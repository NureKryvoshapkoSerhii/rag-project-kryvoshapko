"""Очищення тексту та тематична фільтрація корпусу (Лабораторна робота № 2)."""

from __future__ import annotations

import html
import re
import unicodedata
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Тематичні терміни предметної області Computer Science & IT для відбору статей
# зі знімка Simple English Wikipedia (wikimedia/wikipedia, 20231101.simple).
STRONG_PHRASES: tuple[str, ...] = (
    "computer science",
    "programming language",
    "operating system",
    "data structure",
    "sorting algorithm",
    "search algorithm",
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer network",
    "internet protocol",
    "network protocol",
    "web browser",
    "web server",
    "web application",
    "web development",
    "world wide web",
    "source code",
    "machine code",
    "assembly language",
    "scripting language",
    "markup language",
    "query language",
    "software engineering",
    "software development",
    "computer software",
    "application software",
    "system software",
    "open-source software",
    "open source software",
    "free software",
    "computer program",
    "computer programming",
    "computer hardware",
    "computer memory",
    "computer graphics",
    "computer security",
    "computer virus",
    "computer architecture",
    "central processing unit",
    "graphics processing unit",
    "random-access memory",
    "random access memory",
    "read-only memory",
    "solid-state drive",
    "hard disk drive",
    "file system",
    "file format",
    "database management",
    "relational database",
    "version control",
    "command-line interface",
    "command line interface",
    "graphical user interface",
    "virtual machine",
    "cloud computing",
    "distributed computing",
    "parallel computing",
    "quantum computing",
    "object-oriented programming",
    "functional programming",
    "integrated development environment",
    "hash table",
    "binary tree",
    "binary search",
    "linked list",
    "turing machine",
    "boolean algebra",
    "logic gate",
    "regular expression",
    "cybersecurity",
    "cryptography",
    "encryption",
    "public-key",
    "microprocessor",
    "microcontroller",
    "motherboard",
    "compiler",
    "bytecode",
    "malware",
    "ransomware",
    "spyware",
    "firewall",
    "packet switching",
    "ip address",
    "domain name system",
    "hypertext transfer protocol",
    "transmission control protocol",
    "user datagram protocol",
    "uniform resource locator",
    "application programming interface",
    "character encoding",
    "data compression",
    "garbage collection",
    "multithreading",
    "big data",
    "data mining",
    "computer vision",
)

# Однослівні CS-терміни, що вимагають підтвердження комп'ютерним контекстом
CONTEXT_TOKENS: tuple[str, ...] = (
    "algorithm",
    "software",
    "database",
    "linux",
    "unix",
    "windows",
    "macos",
    "android",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "rust",
    "golang",
    "html",
    "css",
    "sql",
    "http",
    "https",
    "tcp",
    "udp",
    "dns",
    "utf-8",
    "unicode",
    "ascii",
    "json",
    "xml",
    "yaml",
    "git",
    "github",
    "docker",
    "kubernetes",
    "postgresql",
    "mysql",
    "sqlite",
    "mongodb",
    "redis",
    "ethernet",
    "wifi",
    "wi-fi",
    "bluetooth",
    "router",
    "cpu",
    "gpu",
    "ram",
    "bios",
    "firmware",
    "kernel",
    "debugger",
    "parser",
    "lexer",
    "blockchain",
    "cryptocurrency",
    "bitcoin",
    "transistor",
    "semiconductor",
    "server",
    "protocol",
    "internet",
    "computing",
    "programmer",
    "developer",
    "website",
)

CS_ANCHOR_RE = re.compile(
    r"\b(computer|computers|computing|software|hardware|programming|program|"
    r"algorithm|algorithms|database|databases|internet|network|networks|"
    r"operating system|code|coding|web|server|data|digital|processor|cpu|"
    r"browser|encryption|protocol|cryptography|linux|unix)\b",
    re.IGNORECASE,
)

# Виключаємо статті-списки, неоднозначності та нерелевантні біографії/розваги
EXCLUDE_TITLE_RE = re.compile(
    r"(\(disambiguation\)|^list of |\bvideo game\b|\bfilm\b|\bmovie\b|\balbum\b|\bsong\b|\bband\b)",
    re.IGNORECASE,
)

HTML_TAG_HINT_RE = re.compile(r"<[a-zA-Z/][^>]*>|&[a-zA-Z#0-9]+;")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MULTI_SPACES_RE = re.compile(r"[ \t]+")
MULTI_NEWLINES_RE = re.compile(r"\n{3,}")
TOKEN_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {
    tok: re.compile(rf"(?<!\w){re.escape(tok)}(?!\w)", re.IGNORECASE)
    for tok in CONTEXT_TOKENS
}


def is_cs_article(title: str, text: str) -> bool:
    """Перевіряє, чи належить стаття Simple English Wikipedia до тематики CS & IT."""
    if not title or not text:
        return False
    if EXCLUDE_TITLE_RE.search(title):
        return False

    title_l = title.lower()
    lead_l = text[:500].lower()
    combined = f"{title_l}\n{lead_l}"

    # 1. Однозначні терміни предметної області у заголовку або вступному абзаці
    for phrase in STRONG_PHRASES:
        if phrase in combined:
            return True

    # 2. Однослівний ІТ-термін у заголовку + комп'ютерний якір у вступному абзаці
    if CS_ANCHOR_RE.search(lead_l):
        for tok, pattern in TOKEN_BOUNDARY_CACHE.items():
            if pattern.search(title_l):
                return True

    return False


def clean_text(raw_text: str) -> str:
    """Нормалізує кодування у UTF-8, видаляє HTML-розмітку та зайві пробіли."""
    if not raw_text:
        return ""

    # 1. Нормалізація кодування у UTF-8 та Unicode NFC
    text = raw_text.encode("utf-8", errors="ignore").decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    # 2. Видалення базової HTML-розмітки (BeautifulSoup) та HTML-сутностей
    if HTML_TAG_HINT_RE.search(text):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    text = html.unescape(text)

    # 3. Очищення керуючих символів і нормалізація пробілів/переносів рядків
    text = CONTROL_CHARS_RE.sub("", text)
    lines = [MULTI_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = MULTI_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def is_valid_document(text: str, min_char_length: int = 100) -> bool:
    """Фільтрує порожні або надто короткі документи (за замовчуванням < 100 символів)."""
    return len(text.strip()) >= min_char_length
