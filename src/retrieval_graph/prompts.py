"""Default prompts."""

RESPONSE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

<retrieved_docs>
{retrieved_docs}
</retrieved_docs>

System time: {system_time}"""

QUERY_SYSTEM_PROMPT = """Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:

<previous_queries/>
{queries}
</previous_queries>

System time: {system_time}"""

INTENT_SYSTEM_PROMPT = """Determine whether user's most recent query is relevant to the following intent.

<intent_description>
{intent_description}
</intent_description>

System time: {system_time}"""
