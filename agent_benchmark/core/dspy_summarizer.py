"""DSPy signature module for code summarization.

Provides a declarative interface to summarize code files into a high level
summary, key components, and modeling approach guess.

If DSPy or an OpenAI-compatible model isn't available, the caller should fall
back gracefully (handled in analyze module).
"""

try:
    import dspy  # type: ignore
except Exception:  # pragma: no cover
    dspy = None  # type: ignore

if dspy:
    class CodeSummarySignature(dspy.Signature):  # type: ignore
        """Summarize code into a concise description and list important parts.

        code: The source code contents.
        high_level_summary: One or two sentences describing purpose & behavior.
        key_components: Bullet list of notable functions, classes, or steps.
        modeling_approach: Short label for ML/statistical approach if detected.
        """
        code = dspy.InputField()
        high_level_summary = dspy.OutputField(desc="Concise code summary")
        key_components = dspy.OutputField(desc="Bullet list of main components")
        modeling_approach = dspy.OutputField(desc="Modeling technique label")


    class CodeSummarizer(dspy.Module):  # type: ignore
        def __init__(self):
            super().__init__()
            self.gen = dspy.ChainOfThought(CodeSummarySignature)

        def forward(self, code: str):  # type: ignore
            # Light truncation safeguard
            snippet = code[:16000]
            return self.gen(code=snippet)
else:  # pragma: no cover
    class CodeSummarizer:  # type: ignore
        def __init__(self):
            pass
        def __call__(self, code: str):  # mimic DSPy Module call interface
            class R:  # minimal stand-in
                high_level_summary = None
                key_components = None
                modeling_approach = None
            return R()
