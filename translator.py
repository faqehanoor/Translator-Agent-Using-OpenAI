import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables (works locally)
load_dotenv()

# Get API key (works for both local and Streamlit Cloud)
gemini_api_key = os.getenv("GEMINI_API_KEY", default=st.secrets.get("GEMINI_API_KEY", None))

# Check for API key
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found. Add it to your .env file (for local) or Secrets (on Streamlit Cloud).")
    st.stop()

# Setup Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator agent
translator = Agent(
    name="Translator Agent",
    instructions="""You are a translator agent. Translate the given input text into the target language specified by the user. Your response should only contain the translated sentence."""
)

# Async wrapper for Streamlit
async def translate_text(input_text: str):
    return await Runner.run(translator, input=input_text, run_config=config)

# Streamlit UI
st.set_page_config(page_title="Translator Agent", page_icon="🌍")
st.title("🌍 AI Translator Agent (Powered by Gemini)")

input_text = st.text_area("Enter your text with translation instruction:", 
                          "Translate to Urdu: 'The world is changing rapidly with AI.'")

if st.button("Translate"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("🔄 Translating..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(translate_text(input_text))
                st.success("✅ Translation:")
                st.markdown(f"**{result.final_output}**")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
