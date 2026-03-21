import re

def clean_for_tts(text: str) -> str:
    """Clean text before sending to TTS — remove markdown, symbols, URLs."""

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove markdown headers
    text = re.sub(r'#{1,6}\s*', '', text)

    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)

    # Remove backticks and code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove bullet points and list markers
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove special characters that sound bad in TTS
    text = re.sub(r'[*/\\|<>{}[\]^~]', '', text)

    # Remove horizontal rules
    text = re.sub(r'-{3,}', '', text)

    # Replace & with "and"
    text = text.replace('&', ' and ')

    # Replace % with "percent"
    text = text.replace('%', ' percent ')

    # Replace # with nothing (leftover after header removal)
    text = text.replace('#', '')

    # Collapse multiple spaces and blank lines
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()