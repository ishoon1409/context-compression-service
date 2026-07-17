from fastapi import FastAPI
import httpx
import uvicorn

app = FastAPI()

@app.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str):
    
    # 1. REAL GITHUB ISSUE TEST
    if tool_name == "get_github_issue":
        # Let's fetch a real, massive issue from Microsoft's VSCode repository
        url = "https://api.github.com/repos/microsoft/vscode/issues/140000"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            # This returns hundreds of lines of raw, highly bloated JSON!
            return response.json()
            
    # 2. REAL PUBLIC API TEST (Pokemon Data - Very bloated!)
    elif tool_name == "get_web_data":
        url = "https://pokeapi.co/api/v2/pokemon/pikachu"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()

    return {"error": f"Tool '{tool_name}' not found."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)