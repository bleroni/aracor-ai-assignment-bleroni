"""
Module for generating summaries from input text using Large Language Models.
"""

import asyncio
import textwrap

from langchain.schema import HumanMessage
from src.models.model_manager import ModelManager


class SummaryGenerator:
    """
    Generates summaries from input text using a language model.

    Supports generating summaries in various formats:
    - brief: A concise summary (2-3 sentences)
    - detailed: A detailed summary with key points
    - bullet: A bullet point summary
    """

    def __init__(self, model_manager: ModelManager, chunk_size: int = 4000, timeout: int = 30):
        """Initialize SummaryGenerator with ModelManager and configuration."""
        self.model_manager = model_manager
        self.chunk_size = chunk_size  # Max characters per chunk
        self.timeout = timeout  # Seconds before timeout
        self.summary_types = {
            "brief": "Provide a concise summary (2-3 sentences)",
            "detailed": "Provide a detailed summary with key points",
            "bullet": "Provide a summary in bullet point format",
        }

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into manageable chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        for paragraph in text.split("\n\n"):
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

                # Handle very long paragraphs
                while len(current_chunk) > self.chunk_size:
                    split_point = current_chunk.rfind(" ", 0, self.chunk_size)
                    if split_point == -1:
                        split_point = self.chunk_size
                    chunks.append(current_chunk[:split_point])
                    current_chunk = current_chunk[split_point:].strip()

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _summarize_chunk(self, chunk: str, summary_type: str) -> str:
        """Summarize a single chunk with timeout handling."""
        prompt = f"{self.summary_types[summary_type]}:\n\n{chunk}"

        try:
            response = await asyncio.wait_for(
                self.model_manager.default_client.ainvoke([HumanMessage(content=prompt)]), timeout=self.timeout  # pylint: disable=line-too-long  # "black" is reformatting these lines
            )
            return response.content
        except asyncio.TimeoutError:
            return (
                f"[Partial Result - Timeout after {self.timeout}s]: " f"{textwrap.shorten(chunk, width=100, placeholder='...')}"  # pylint: disable=line-too-long  # "black" is reformatting these lines
            )

    async def _summarize_chunks(self, chunks: list[str], summary_type: str) -> list[str]:
        """Process all chunks concurrently."""
        tasks = [self._summarize_chunk(chunk, summary_type) for chunk in chunks]
        return await asyncio.gather(*tasks)

    def generate_summary(self, text: str, summary_type: str = "brief") -> str:
        """Generate summary of the input text with specified type."""
        if summary_type not in self.summary_types:
            raise ValueError(f"Invalid summary type. " f"Choose from: " f"{list(self.summary_types.keys())}")  # pylint: disable=line-too-long  # "black" is reformatting these lines

        # Chunk the text
        chunks = self._chunk_text(text)

        # If single chunk, process synchronously for simplicity
        if len(chunks) == 1:
            response = self.model_manager.default_client.invoke(
                [HumanMessage(content=f"{self.summary_types[summary_type]}:\n\n{chunks[0]}")]
            )
            return response.content

        # Process multiple chunks asynchronously
        summaries = asyncio.run(self._summarize_chunks(chunks, summary_type))

        # Combine results based on summary type
        if summary_type == "bullet":
            combined = "\n".join(summaries)
        else:
            combined = " ".join(summaries)

        return combined

    def set_model(self, model_name: str):
        """Switch the underlying model."""
        self.model_manager.switch_client(model_name)


# Example usage
if __name__ == "__main__":
    # Initialize ModelManager and SummaryGenerator
    model_mgr = ModelManager()
    summarizer = SummaryGenerator(model_mgr)

    # Sample text (replace with your own)
    SAMPLE_TEXT = """
    Artificial Intelligence (AI) is transforming various industries. 
    It enables machines to perform tasks that typically require human intelligence.

    In healthcare, AI assists in diagnosis and treatment planning. 
    In finance, it helps detect fraud and optimize trading strategies.

    The technology continues to evolve rapidly, raising both opportunities 
    and challenges for the future.
    """

    # Generate different types of summaries
    try:
        # Brief summary
        brief = summarizer.generate_summary(SAMPLE_TEXT, "brief")
        print("Brief Summary:")
        print(brief)
        print("\n")

        # Detailed summary
        detailed = summarizer.generate_summary(SAMPLE_TEXT, "detailed")
        print("Detailed Summary:")
        print(detailed)
        print("\n")

        # Bullet point summary
        bullet = summarizer.generate_summary(SAMPLE_TEXT, "bullet")
        print("Bullet Point Summary:")
        print(bullet)

        # Switch model and try again
        summarizer.set_model("cohere")
        print("\nSwitching to Cohere model:")
        brief_cohere = summarizer.generate_summary(SAMPLE_TEXT, "brief")
        print(brief_cohere)

    except Exception as e:  # pylint: disable=W0718
        print(f"Error: {str(e)}")
