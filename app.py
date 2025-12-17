from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from bot import MyAgentBot
from main import OpenAIAgent
import os
import traceback
import uuid

# In-memory store for A2A tasks so orchestrators that poll `tasks/get` can retrieve
# the final result.
A2A_TASKS: dict[str, dict] = {}

# Bot Framework Adapter settings
SETTINGS = BotFrameworkAdapterSettings(
    app_id=os.getenv("MICROSOFT_APP_ID", ""),
    app_password=os.getenv("MICROSOFT_APP_PASSWORD", "")
)

# Create adapter
ADAPTER = BotFrameworkAdapter(SETTINGS)

# Create your OpenAI agent
openai_agent = OpenAIAgent()

# Create bot
BOT = MyAgentBot(openai_agent)

# Error handler
async def on_error(context, error):
    print(f"❌ Error: {error}", flush=True)
    print(traceback.format_exc())
    try:
        await context.send_activity("Sorry, an error occurred.")
    except:
        pass

ADAPTER.on_turn_error = on_error

# Handle incoming messages
async def messages(req: web.Request) -> web.Response:
    try:
        print(f"\n📥 Incoming request from {req.remote}")
        
        # Check content type
        if req.content_type != "application/json" and "application/json" not in req.content_type:
            print(f"❌ Invalid content type: {req.content_type}")
            return web.Response(status=415, text="Content-Type must be application/json")

        # Parse the JSON body
        body = await req.json()
        
        # Check if this is a JSON-RPC (A2A) message
        if "jsonrpc" in body and body.get("method"):
            method = body.get("method")
            print(f"📨 Detected JSON-RPC (Agent-to-Agent) message: {method}")

            if method == "message/send":
                return await handle_a2a_message(body)
            if method == "tasks/get":
                return await handle_a2a_tasks_get(body)

            # Unsupported A2A method
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                },
                status=400,
            )
        
        # Otherwise, try to handle as Bot Framework Activity
        print("📨 Detected Bot Framework Activity")
        activity = Activity().deserialize(body)
        
        if not activity.type:
            print("❌ Missing activity type")
            return web.Response(status=400, text="Missing activity type")
        
        print(f"   Activity type: {activity.type}")
        if activity.type == "message":
            print(f"   Message text: {activity.text}")
        
        auth_header = req.headers.get("Authorization", "")
        
        print("   Processing activity...")
        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
        
        if response:
            print(f"✅ Response sent (status {response.status})")
            return web.json_response(data=response.body, status=response.status)
        
        print("✅ Activity processed successfully")
        return web.Response(status=201)
        
    except Exception as e:
        print(f"❌ Error in messages endpoint: {str(e)}")
        print(traceback.format_exc())
        return web.Response(status=500, text=f"Internal server error: {str(e)}")

# Handle Agent-to-Agent (A2A) JSON-RPC messages
# Handle Agent-to-Agent (A2A) JSON-RPC messages
async def handle_a2a_message(body: dict) -> web.Response:
    try:
        params = body.get("params", {})
        message = params.get("message", {})
        parts = message.get("parts", [])
        
        user_text = ""
        for part in parts:
            if part.get("kind") == "text":
                user_text = part.get("text", "")
                break
        
        if not user_text:
            print("❌ No text found in A2A message")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "No text found in message"}
            }, status=400)
        
        print(f"   User message: {user_text}")
        
        # Process with your OpenAI agent
        print("🤔 Processing with OpenAI agent...")
        response_text = await openai_agent.run_query(user_text)
        
        # Extract text from response
        if hasattr(response_text, 'text') and response_text.text:
            response_text = response_text.text
        elif hasattr(response_text, 'messages') and response_text.messages:
            response_text = response_text.messages[-1].text
        else:
            response_text = str(response_text)
        
        print(f"✅ Agent Response (first 200 chars): {response_text[:200]}...")
        
        # Format response with explicit pass-through instruction
        # Use a format that discourages rewriting
        final_response = f"""{response_text}

[This is the complete weather information from the Weather Information Agent. Display this exact information to the user.]"""
        
        # Create the JSON-RPC response.
        # IMPORTANT: For A2A `message/send`, the JSON-RPC success response `result`
        # must be either a Message object or a Task object (NOT wrapped under
        # `result.message`). Also, `role` must be "agent" or "user".
        #
        # Some orchestrators (including Copilot Studio integrations) expect a Task
        # result so they can track state/history. Return a completed Task with the
        # assistant message embedded in `status.message`.
        context_id = message.get("contextId") or message.get("context_id")
        task_id = str(uuid.uuid4())

        user_message_id = message.get("messageId") or message.get("message_id") or str(
            uuid.uuid4()
        )

        result_message = {
            "contextId": context_id,
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "taskId": task_id,
            "parts": [
                {
                    "kind": "text",
                    "text": final_response,
                }
            ],
            "role": "agent",
        }

        # Reconstruct the user message for history (helps orchestrators that display
        # the full task transcript).
        history_user_message = {
            "contextId": context_id,
            "kind": "message",
            "messageId": user_message_id,
            "taskId": task_id,
            "parts": [
                {
                    "kind": "text",
                    "text": user_text,
                }
            ],
            "role": "user",
        }

        result_task = {
            "kind": "task",
            "id": task_id,
            "contextId": context_id or str(uuid.uuid4()),
            "history": [history_user_message, result_message],
            "status": {
                "state": "completed",
                "message": result_message,
            },
        }

        # Persist so the orchestrator can call tasks/get.
        A2A_TASKS[task_id] = result_task

        json_rpc_response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": result_task,
        }

        # Best-effort schema validation for easier debugging in logs.
        try:
            from a2a.types import SendMessageSuccessResponse

            SendMessageSuccessResponse.model_validate(json_rpc_response)
        except Exception as ve:
            print(f"❌ A2A response validation failed: {ve}")
            # Still return the response; the caller may provide additional diagnostics.
        
        print(f"📤 Sending response ({len(final_response)} chars)")

        return web.json_response(json_rpc_response)
        
    except Exception as e:
        print(f"❌ Error handling A2A message: {str(e)}")
        print(traceback.format_exc())
        return web.json_response({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        }, status=500)


async def handle_a2a_tasks_get(body: dict) -> web.Response:
    """Handle A2A `tasks/get` so orchestrators can poll for the result."""
    try:
        params = body.get("params", {})
        task_id = params.get("id")
        if not task_id:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32602,
                        "message": "Invalid parameters: missing params.id",
                    },
                },
                status=400,
            )

        task = A2A_TASKS.get(task_id)
        if not task:
            # A2A TaskNotFoundError code
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32001,
                        "message": "Task not found",
                        "data": {"id": task_id},
                    },
                },
                status=404,
            )

        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": task,
            }
        )

    except Exception as e:
        print(f"❌ Error handling tasks/get: {str(e)}")
        print(traceback.format_exc())
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            },
            status=500,
        )

# Health check endpoint
async def health(req: web.Request) -> web.Response:
    return web.json_response({"status": "healthy", "agent": openai_agent.agent_name})

# Create web app
app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "3978"))
        print(f"🚀 Starting bot server on port {port}...")
        print(f"📍 Endpoint: http://localhost:{port}/api/messages")
        print(f"💚 Health check: http://localhost:{port}/health")
        print(f"🔗 Supports: Bot Framework Activity & Agent-to-Agent (JSON-RPC)")
        print("\n✅ Server ready!\n")
        web.run_app(app, host="0.0.0.0", port=port)
    except Exception as error:
        print(f"❌ Failed to start server: {error}")
        raise error
