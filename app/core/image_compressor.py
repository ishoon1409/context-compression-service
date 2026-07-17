import base64
import math
import os
import re
import tiktoken
import httpx
import io
import asyncio
from PIL import Image
from dotenv import load_dotenv

# Force load the .env file to guarantee the API key is found
load_dotenv()

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Qwen3-VL tokenizes images using 32x32 pixel patches (Qwen2.5-VL used 28x28 — this changed in v3).
QWEN3_VL_PATCH_SIZE = 32


def strip_reasoning_trace(text: str) -> str:
    """Removes any <think>...</think> reasoning block a thinking-tuned model may emit,
    plus any leftover leading/trailing whitespace from the removal."""
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    return cleaned.strip()


def calculate_qwen3_vl_image_tokens(width: int, height: int) -> int:
    """
    Estimates the number of visual tokens Qwen3-VL will consume for an image of
    the given pixel dimensions, based on its 32x32 patch-based tokenization scheme.
    This is the actual 'before' cost — what the model is charged for the raw image
    itself, as opposed to a generic base64-as-text approximation.
    """
    patches_wide = math.ceil(width / QWEN3_VL_PATCH_SIZE)
    patches_high = math.ceil(height / QWEN3_VL_PATCH_SIZE)
    return patches_wide * patches_high


async def compress_image_semantic(file_bytes: bytes, filename: str = "upload.jpg", mime_type: str = "image/jpeg") -> dict:
    """
    Takes raw image bytes, converts them to Base64, and uses the Qwen-VL model 
    on Neysa via a raw HTTPX request to translate visual pixels into a dense semantic summary.
    """
    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode('utf-8')

    # Capture original upload dimensions BEFORE any resizing, since that's what
    # LibreChat would have sent to the provider had this proxy not intervened.
    try:
        original_dims_image = Image.open(io.BytesIO(file_bytes))
        original_width, original_height = original_dims_image.size
        # DEBUG: confirms what this service actually received. If this doesn't match
        # the source file's true resolution, LibreChat is resizing client-side before upload.
        print(f"[IMAGE-COMPRESSOR] Received image dimensions: {original_width}x{original_height}")
    except Exception as local_img_err:
        raise Exception(f"The uploaded file is corrupted or not a valid image. Local parser failed: {str(local_img_err)}")

    # --- ADAPTIVE RESOLUTION RETRY LOOP ---
    # If the 235B model times out (504), we automatically shrink the image
    # and retry. We start at 256x256 to ensure absolute minimum GPU memory usage.
    resolutions = [(256, 256), (192, 192), (128, 128)]
    last_error = None
    
    for attempt, max_size in enumerate(resolutions):
        try:
            # 1. Normalize the image locally first!
            try:
                # Open the image locally and force it into a pristine standard JPEG
                image = Image.open(io.BytesIO(file_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Dynamically resize based on the current attempt
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                buffer = io.BytesIO()
                # Save with 85% quality to further reduce the Base64 payload size
                image.save(buffer, format="JPEG", quality=85)
                clean_jpeg_bytes = buffer.getvalue()
            except Exception as local_img_err:
                raise Exception(f"The uploaded file is corrupted or not a valid image. Local parser failed: {str(local_img_err)}")

            base64_image = base64.b64encode(clean_jpeg_bytes).decode('utf-8')
            mime_type = "image/jpeg"

            # 2. The Semantic Translation Prompt
            prompt = (
                "You are an expert visual data extraction engine. "
                "Analyze this image and convert all visual information, embedded text (OCR), charts, "
                "and spatial relationships into a highly dense, compressed bullet list "
                "organized under exactly these four headers, in this order:\n\n"
                "Subject: who or what is the main focus of the image.\n"
                "Setting: the location, background, and environmental context.\n"
                "Actions: what is happening or being done in the image.\n"
                "Notable details: embedded text (OCR), colors, spatial relationships, "
                "and any other specific factual details worth preserving.\n\n"
                "FORMATTING RULES: Use each header exactly once, followed by hyphen (-) bullet points. "
                "Absolutely DO NOT use asterisks (*) anywhere in your output. "
                "Use double underscores (__text__) or HTML tags (<b>text</b>) to make text bold. "
                "Do not include any conversational filler. Output ONLY the four headers and their bullets. "
                "CRITICAL INSTRUCTION: Do NOT output long reasoning or thinking steps. Answer immediately and concisely."
            )

            # 3. Construct the Raw Request
            url = "https://aistudio-inference.neysa.io/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('NEYSA_API_KEY')}"
            }
            
            payload = {
                "model": "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
                "user": "ishaan",  # REQUIRED BY NEYSA: The user tracking field
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "low" # Force low detail mode
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 512  # FORCE the model to finish quickly and prevent endless loops
            }

            # 4. Call the Neysa Endpoint using pure HTTPX
            print(f"Routing {filename} (Resized to {max_size[0]}x{max_size[1]}) to Qwen-VL via pure HTTP request...")
            
            # Using 120 seconds timeout so the gateway doesn't kill the connection silently
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                # Catch Timeouts/Bad Gateways and trigger the retry loop
                if response.status_code in [504, 502, 429]:
                    last_error = f"Neysa API Error {response.status_code}: {response.text}"
                    print(f"Attempt {attempt + 1} failed ({response.status_code}). Retrying with smaller image...")
                    await asyncio.sleep(2)
                    continue
                elif response.status_code != 200:
                    raise Exception(f"Neysa API Error {response.status_code}: {response.text}")
                    
                response_data = response.json()
                raw_caption = response_data["choices"][0]["message"]["content"].strip()
                semantic_caption = strip_reasoning_trace(raw_caption)

            # 5. Calculate REAL Before/After Token Metrics
            # "Before": what Qwen3-VL would have charged for the image AS RECEIVED
            # (post any LibreChat client-side resizing), using its 32x32 patch formula.
            original_tokens = calculate_qwen3_vl_image_tokens(original_width, original_height)

            tokenizer = tiktoken.get_encoding("cl100k_base")
            compressed_tokens = len(tokenizer.encode(semantic_caption))

            ratio = round((1 - (compressed_tokens / original_tokens)) * 100) if original_tokens else 0
            ratio = max(0, ratio)

            # If successful, break the loop and return!
            return {
                "semantic_caption": semantic_caption,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": ratio
            }

        except Exception as e:
            last_error = str(e)
            # If it's a local file error, don't bother retrying. Throw immediately.
            if "Local parser failed" in last_error:
                raise e
            # Otherwise, allow the loop to try the next resolution

    # If all 3 attempts fail, throw the final error
    raise Exception(f"Image Compression via Neysa failed after 3 attempts. Last error: {last_error}")