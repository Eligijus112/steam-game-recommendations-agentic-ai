from mcp_servers.app import mcp


@mcp.resource("resource://vector-db/metadata")
def vector_db_information() -> str:
    """Describe the game vector store so the LLM can build effective queries.

    Explains what is embedded (used for semantic search) and which payload
    fields are indexed for metadata sub-filtering, including their types and the
    kind of filter each supports.
    """
    return """
        Vector store: Steam games catalog.

        Vector store provider: Qdrant

        Semantic search field (embedded as the vector):
        - detailed_description: the long-form game description. Phrase semantic
          queries in terms of gameplay, themes, setting, and mood.

        Indexed metadata for sub-filtering (fast, exact filters):
        - genre (string): single primary genre, e.g. "Action".
        - genres (string array): all genres; matches if ANY element equals the
          value, e.g. "Indie", "RPG".
        - tags (string array): Steam user tags; matches if ANY element equals the
          value, e.g. "Co-op", "Open World", "Singleplayer".
        - developer (string): studio name, exact match.
        - publisher (string): publisher name, exact match.
        - is_free (boolean): True for free-to-play titles, False otherwise.
        - price_usd (float): price in USD; supports range filters (e.g. <= 20).
        - rating (float): RAWG rating from 0 to 5; supports range filters.
        - metacritic (float): Metacritic score from 0 to 100; supports range
          filters.

        Other payload fields are returned with each result but are NOT indexed,
        so prefer the fields above for filtering: name, steam_appid, rawg_id,
        released, rating_top, ratings_count, added, playtime, website, steam_url,
        short_description, steam_short_description.
    """
