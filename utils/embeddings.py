from openai import AsyncOpenAI

# OpenAI client for embeddings
client = AsyncOpenAI()

async def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding 