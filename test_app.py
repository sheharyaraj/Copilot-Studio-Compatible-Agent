from aiohttp import web
import traceback
import sys

async def messages(req):
    try:
        print("\n" + "="*60)
        print("📥 REQUEST RECEIVED")
        print("="*60)
        
        # Log everything about the request
        print(f"Method: {req.method}")
        print(f"Content-Type: {req.content_type}")
        print(f"Headers: {dict(req.headers)}")
        
        # Try to read the body
        try:
            body = await req.json()
            print(f"Body received: {body}")
        except Exception as e:
            print(f"❌ Failed to parse JSON: {e}")
            return web.Response(status=400, text=f"Invalid JSON: {e}")
        
        # Try to import and use your agent
        try:
            print("\n🔄 Importing OpenAI agent...")
            from main import OpenAIAgent
            print("✅ Import successful")
            
            print("🔄 Initializing agent...")
            agent = OpenAIAgent()
            print("✅ Agent initialized")
            
            # Get the message text
            user_text = body.get('text', '')
            print(f"\n💬 User message: {user_text}")
            
            if not user_text:
                return web.json_response({"error": "No text in message"}, status=400)
            
            # Run the query
            print("🔄 Running query...")
            import asyncio
            response = await agent.run_query(user_text)
            print(f"✅ Got response: {response[:100]}...")
            
            return web.json_response({
                "type": "message",
                "text": response
            })
            
        except Exception as e:
            print(f"\n❌ ERROR IN AGENT: {e}")
            print(traceback.format_exc())
            return web.Response(status=500, text=f"Agent error: {e}")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print(traceback.format_exc())
        return web.Response(status=500, text=f"Server error: {e}")

async def health(req):
    return web.json_response({"status": "ok"})

app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    print("🚀 Starting test server on port 3978...")
    print("📍 Endpoints:")
    print("   POST /api/messages")
    print("   GET  /health")
    print("\n✅ Server ready!\n")
    web.run_app(app, host="0.0.0.0", port=3978)