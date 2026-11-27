import json

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def json_object(text):
    try:
        result = json.loads(text or "")
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


async def run_workflow(agent, client, model, query, state):
    if "client" in Gemini.model_fields:
        gemini = Gemini(model=model, client=client)
    else:
        # ADK 1.22 exposes its client through a cached property.
        gemini = Gemini(model=model)
        gemini.api_client = client

    def bind_model(member):
        if isinstance(member, LlmAgent):
            member.model = gemini
        for child in member.sub_agents:
            bind_model(child)

    bind_model(agent)
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name="agents", user_id="researcher", state={"query": query, **state}
    )
    runner = Runner(agent=agent, app_name=session.app_name, session_service=sessions)
    print(f"   ADK Runner executing {agent.name}")
    async for _ in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        pass
    session = await sessions.get_session(
        app_name=session.app_name, user_id=session.user_id, session_id=session.id
    )
    return session.state
