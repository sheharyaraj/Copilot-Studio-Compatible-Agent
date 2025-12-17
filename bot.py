from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
import traceback

class MyAgentBot(ActivityHandler):
    def __init__(self, openai_agent):
        self.agent = openai_agent
        print("✅ Bot initialized with OpenAI agent")
    
    async def on_message_activity(self, turn_context: TurnContext):
        try:
            print(f"📨 Received message: {turn_context.activity.text}")
            
            # Get user message
            user_message = turn_context.activity.text
            
            if not user_message:
                await turn_context.send_activity("Please send a message.")
                return
            
            # Process with your existing agent
            print("🤔 Processing with OpenAI agent...")
            response = await self.agent.run_query(user_message)
            
            # Extract text from AgentRunResponse
            if hasattr(response, 'text') and response.text:
                # Use the top-level text attribute if available
                response_text = response.text
            elif hasattr(response, 'messages') and response.messages:
                # Otherwise get text from the last message
                response_text = response.messages[-1].text
            else:
                # Fallback
                response_text = str(response)
            
            print(f"✅ Sending response: {response_text[:100]}...")
            
            # Send response back
            await turn_context.send_activity(str(response_text))
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in on_message_activity: {error_msg}")
            
            # Don't show serviceUrl errors to the user (they're expected in testing)
            if "test.com" not in error_msg:
                print(traceback.format_exc())
                await turn_context.send_activity(f"Sorry, I encountered an error: {error_msg}")
    
    async def on_members_added_activity(
        self, members_added: list[ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    f"Hello! I'm {self.agent.agent_name}. How can I help you?"
                )
