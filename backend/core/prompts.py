"""
backend/core/prompts.py
───────────────────────
All LLM prompt templates for every writing mode and intensity level.

Design principles:
  - System prompts are precise and rule-based.
  - User prompts inject the analysed text with preservation lists.
  - Mode-specific instructions shape the register, vocabulary, and rhythm.
  - Preservation rules are listed explicitly and checked post-generation.
"""
from __future__ import annotations

from backend.core.schemas import (
    EnglishVariant,
    Intensity,
    TextAnalysis,
    WritingMode,
)


# ── Mode-specific instructions ────────────────────────────────────────────────

_MODE_INSTRUCTIONS: dict[WritingMode, str] = {

    WritingMode.NATURAL: """\
Write as a thoughtful, articulate person would in a blog post, article,
or personal essay.  Sentences should vary in length and rhythm – mix
short punchy sentences with longer flowing ones.  Use contractions where
natural (it's, don't, we've).  Avoid passive voice unless necessary.
Sound like a real person, not a report generator.""",

    WritingMode.ACADEMIC: """\
Write in scholarly but accessible academic prose.  Use discipline-specific
terminology where appropriate, but do not over-complicate simple ideas.
Maintain an objective tone, use hedging language where claims are not
absolute (may, suggests, appears to), and ensure each paragraph builds
logically on the last.  Do not use contractions.  Avoid colloquialisms.""",

    WritingMode.PROFESSIONAL: """\
Write in polished workplace English – clear, direct, and confident.
Use active voice.  Keep paragraphs focused.  Avoid bureaucratic
jargon (utilize → use, leverage → use).  Strike a tone that is
professional but warm, not cold or robotic.  Bullet points are acceptable
where they aid readability.""",

    WritingMode.SIMPLE: """\
Write in plain English for a wide general audience (reading age ~12).
Use short sentences (< 20 words where possible).  Prefer common, everyday
words.  Avoid technical jargon unless it cannot be avoided – and if you
use it, explain it briefly.  Break long paragraphs into shorter ones.
Use active voice throughout.""",

    WritingMode.RESEARCH: """\
Write in formal research prose suitable for a journal article,
dissertation, or technical report.  Use precise language, avoid ambiguity,
and maintain a consistently objective tone.  Include hedging language for
claims not supported by the provided data.  Structure paragraphs around a
single clear point.  Preserve all citation and reference formatting exactly.""",

    WritingMode.CONVERSATIONAL: """\
Write as if speaking naturally to someone in a podcast, interview, or
informal meeting.  Use contractions freely.  Rhetorical questions are
welcome.  Vary sentence length dramatically – some very short, some longer.
Use em-dashes and ellipses for natural pauses.  First and second person
(I, we, you) is encouraged.  Avoid any trace of corporate or academic
stiffness.""",

    WritingMode.PERSONAL: """\
Preserve the author's unique voice as much as possible.  Focus on
removing the AI patterns (robotic openers, clichéd transitions, repetitive
structures) without changing the personality or emotional tone of the
writing.  Do not impose a new style – refine and clarify the existing one.
If the author uses informal punctuation, light humour, or personal
expressions, keep them.""",
}


# ── Intensity modifiers ───────────────────────────────────────────────────────

_INTENSITY_INSTRUCTIONS: dict[Intensity, str] = {
    Intensity.LIGHT: """\
Transformation intensity: LIGHT.
Make minimal changes.  Fix only the most obvious AI patterns, awkward
phrases, and repetitive structures.  Preserve the original structure and
most of the original wording.  Do not restructure paragraphs.""",

    Intensity.BALANCED: """\
Transformation intensity: BALANCED.
Rework sentences and transitions where needed.  Restructure some paragraphs
if flow requires it.  Remove clichéd and robotic phrases.  Keep the core
argument and structure intact, but make the prose feel naturally written.""",

    Intensity.STRONG: """\
Transformation intensity: STRONG.
Substantially rewrite the text to remove all AI patterns.  Reorganise
sentences and paragraphs for maximum natural flow and readability.  You may
merge or split sentences, change paragraph order where logical, and
introduce varied sentence rhythms.  The result should read as if written
from scratch by an expert human author in this mode.  Do not add new
facts, but do express existing ideas freshly.""",
}


# ── Preservation instruction builder ─────────────────────────────────────────

def _build_preservation_block(
    analysis: TextAnalysis,
    preserve_citations: bool,
    preserve_technical_terms: bool,
    preserve_numbers: bool = True,
) -> str:
    lines: list[str] = [
        "═══ PRESERVATION RULES (MUST follow exactly) ═══",
        "• Never change any numerical value, percentage, or statistic.",
        "• Never alter author names, publication years, or place names.",
        "• Never invent new facts, examples, or references.",
    ]

    if preserve_citations and analysis.citations:
        lines.append("• Preserve ALL citations exactly as written:")
        for c in analysis.citations[:20]:
            lines.append(f"    – {c}")

    if preserve_technical_terms and analysis.technical_terms:
        lines.append("• Preserve these technical terms (do not simplify or replace):")
        for t in analysis.technical_terms[:30]:
            lines.append(f"    – {t}")

    if analysis.numerical_values:
        lines.append("• These numerical values must appear unchanged in the output:")
        lines.append(f"    {', '.join(analysis.numerical_values[:20])}")

    lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(lines)


# ── Issue summary ─────────────────────────────────────────────────────────────

def _build_issues_summary(analysis: TextAnalysis) -> str:
    lines = ["The following AI-writing problems were detected – fix them all:"]

    if analysis.repetitive_structures:
        lines.append(f"• Repetitive sentence structures: "
                     f"{', '.join(r.phrase for r in analysis.repetitive_structures[:5])}")

    if analysis.excessive_transitions:
        lines.append(f"• Overused transitions: {', '.join(analysis.excessive_transitions[:5])}")

    if analysis.unnatural_phrases:
        lines.append(f"• Unnatural/robotic phrases: {', '.join(analysis.unnatural_phrases[:5])}")

    if analysis.overly_formal_expressions:
        lines.append(f"• Overly formal expressions: {', '.join(analysis.overly_formal_expressions[:5])}")

    if analysis.repetitive_openers:
        lines.append(f"• Repetitive/generic sentence openers detected.")

    if analysis.redundant_explanations:
        lines.append(f"• Redundant explanations detected – merge or remove them.")

    if analysis.weak_connections:
        lines.append(f"• Weak inter-sentence connections – improve flow.")

    if len(lines) == 1:
        lines.append(
            "• No specific problems were auto-detected, but the text still "
            "reads like machine-generated prose. Rewrite it fully so it sounds "
            "naturally human."
        )

    return "\n".join(lines)


# ── Main prompt builders ──────────────────────────────────────────────────────

SYSTEM_BASE = """\
You are an expert writing editor and linguist specialising in transforming
AI-generated text into natural, fluent, human-sounding prose.

Your sole task is to REWRITE the user's text according to the given mode,
intensity, and rules.  Return ONLY the rewritten text – no preamble, no
explanation, no commentary, no markdown fences."""


def build_system_prompt(
    mode: WritingMode,
    intensity: Intensity,
    english_variant: EnglishVariant,
) -> str:
    variant_note = (
        "Use American English spelling (color, recognize, analyze)."
        if english_variant == EnglishVariant.AMERICAN
        else "Use British English spelling (colour, recognise, analyse)."
    )

    return "\n\n".join([
        SYSTEM_BASE,
        f"Writing Mode:\n{_MODE_INSTRUCTIONS[mode]}",
        f"{_INTENSITY_INSTRUCTIONS[intensity]}",
        variant_note,
        "CRITICAL: Output ONLY the rewritten text.  Do not explain, comment, or add headers.",
    ])


def build_user_prompt(
    text: str,
    analysis: TextAnalysis,
    preserve_citations: bool,
    preserve_technical_terms: bool,
) -> str:
    preservation_block = _build_preservation_block(
        analysis, preserve_citations, preserve_technical_terms
    )
    issues_summary = _build_issues_summary(analysis)

    return "\n\n".join([
        "Rewrite the text below so it reads like natural human writing. "
        "Do not return the input unchanged: rephrase sentences, vary the "
        "rhythm and word choice, and improve flow throughout. Copying the "
        "original wording is not an acceptable result.",
        issues_summary,
        preservation_block,
        "Placeholders such as ⟦CITE_0001⟧, ⟦TERM_0001⟧ and ⟦NUM_0001⟧ must "
        "be kept exactly as written – they are restored automatically.",
        "TEXT TO REWRITE (between the markers):",
        "<<<BEGIN>>>",
        text.strip(),
        "<<<END>>>",
        "Return ONLY the rewritten text.  Preserve paragraph breaks.",
    ])


# ── Sentence-level prompt ─────────────────────────────────────────────────────

def build_sentence_prompt(
    sentence: str,
    context_before: str,
    context_after: str,
    mode: WritingMode,
    intensity: Intensity,
) -> str:
    """Used when granularity = SENTENCE.  Returns a focused rewrite prompt."""
    return "\n\n".join([
        f"Rewrite only the MARKED sentence to sound natural and human "
        f"(mode={mode.value}, intensity={intensity.value}).",
        f"Context before: {context_before[-200:]}",
        f"[REWRITE THIS] {sentence}",
        f"Context after: {context_after[:200]}",
        "Return ONLY the rewritten sentence – nothing else.",
    ])
