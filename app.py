"""
Task-03: Text Generation with Markov Chains
Implements both character-level and word-level Markov chain models with Gradio UI.
"""

import random
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import gradio as gr

# ── Sample corpora ─────────────────────────────────────────────────────────
CORPORA = {
    "Shakespeare": """
To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heartache and the thousand natural shocks that flesh is heir to tis a consummation
devoutly to be wished to die to sleep to sleep perchance to dream ay there's the rub
for in that sleep of death what dreams may come when we have shuffled off this mortal coil
must give us pause there's the respect that makes calamity of so long life
    """.strip(),

    "Tech Blog": """
Artificial intelligence is transforming every industry in the modern world.
Machine learning algorithms can predict outcomes with remarkable accuracy.
Deep learning neural networks have revolutionized computer vision and NLP.
The future of AI lies in creating systems that can reason and learn.
Generative models are capable of producing realistic images text and audio.
Transfer learning allows models to be adapted for specific domain tasks.
The transformer architecture has become the foundation of modern NLP.
Attention mechanisms allow models to focus on relevant parts of the input.
Large language models have demonstrated emergent capabilities at scale.
Prompt engineering is a new discipline for effectively communicating with AI.
    """.strip(),

    "Nature Writing": """
The mountains rose above the valley like ancient sentinels watching over the land.
Rivers carved their paths through rock and earth over countless millennia.
Forest canopies sheltered countless creatures from rain and harsh sunlight.
The ocean tides moved in rhythms older than human memory or recorded history.
Seasons changed with a slow certainty that grounded every living thing.
Birds migrated across continents following instincts older than civilization.
The stars wheeled overhead reminding us of the vast scale of cosmic time.
Morning dew settled on leaves and petals in the quiet before the world awoke.
Wind moved through tall grass in waves that looked like breathing and thinking.
    """.strip(),
}


# ── Word-Level Markov Chain ────────────────────────────────────────────────
class WordMarkovChain:
    def __init__(self, order: int = 2):
        self.order = order
        self.model: Dict[Tuple, List[str]] = defaultdict(list)
        self.start_states: List[Tuple] = []

    def train(self, text: str) -> None:
        words = text.lower().split()
        if len(words) <= self.order:
            raise ValueError("Text too short for given order.")

        self.model.clear()
        self.start_states.clear()

        for i in range(len(words) - self.order):
            state = tuple(words[i : i + self.order])
            next_word = words[i + self.order]
            self.model[state].append(next_word)

        # Collect start states (beginning of sentences)
        sentences = re.split(r'[.!?]+', text.lower())
        for sent in sentences:
            words_s = sent.split()
            if len(words_s) >= self.order:
                self.start_states.append(tuple(words_s[:self.order]))

    def generate(self, seed: str = "", max_words: int = 100) -> str:
        if not self.model:
            raise RuntimeError("Model not trained yet.")

        # Pick start state
        if seed:
            seed_words = seed.lower().split()
            state = tuple(seed_words[-self.order:]) if len(seed_words) >= self.order else None
            if state not in self.model:
                state = None
        else:
            state = None

        if state is None:
            state = random.choice(self.start_states) if self.start_states else random.choice(list(self.model.keys()))

        result = list(state)

        for _ in range(max_words - self.order):
            candidates = self.model.get(state)
            if not candidates:
                break
            next_word = random.choice(candidates)
            result.append(next_word)
            state = tuple(result[-self.order:])

        return " ".join(result).capitalize()

    def stats(self) -> dict:
        return {
            "States (n-grams)": len(self.model),
            "Order": self.order,
            "Total transitions": sum(len(v) for v in self.model.values()),
            "Avg. transitions/state": round(
                sum(len(v) for v in self.model.values()) / max(len(self.model), 1), 2
            ),
        }


# ── Character-Level Markov Chain ───────────────────────────────────────────
class CharMarkovChain:
    def __init__(self, order: int = 4):
        self.order = order
        self.model: Dict[str, List[str]] = defaultdict(list)

    def train(self, text: str) -> None:
        self.model.clear()
        for i in range(len(text) - self.order):
            state = text[i : i + self.order]
            next_char = text[i + self.order]
            self.model[state].append(next_char)

    def generate(self, seed: str = "", max_chars: int = 500) -> str:
        if not self.model:
            raise RuntimeError("Model not trained yet.")

        keys = list(self.model.keys())
        if seed and len(seed) >= self.order:
            state = seed[-self.order:]
        else:
            state = random.choice(keys)

        result = list(state)

        for _ in range(max_chars - self.order):
            candidates = self.model.get(state)
            if not candidates:
                # pick a random known state
                state = random.choice(keys)
                continue
            next_char = random.choice(candidates)
            result.append(next_char)
            state = "".join(result[-self.order:])

        return "".join(result)

    def stats(self) -> dict:
        return {
            "States": len(self.model),
            "Order": self.order,
            "Total transitions": sum(len(v) for v in self.model.values()),
        }


# ── Global model instances ─────────────────────────────────────────────────
word_model = WordMarkovChain(order=2)
char_model = CharMarkovChain(order=4)


def train_word_model(corpus_choice: str, custom_text: str, order: int) -> str:
    global word_model
    text = custom_text.strip() if custom_text.strip() else CORPORA.get(corpus_choice, "")
    if not text:
        return "⚠️ No text provided."
    word_model = WordMarkovChain(order=int(order))
    word_model.train(text)
    stats = word_model.stats()
    return (
        f"✅ Word-level model trained!\n\n"
        + "\n".join(f"  {k}: {v}" for k, v in stats.items())
    )


def train_char_model(corpus_choice: str, custom_text: str, order: int) -> str:
    global char_model
    text = custom_text.strip() if custom_text.strip() else CORPORA.get(corpus_choice, "")
    if not text:
        return "⚠️ No text provided."
    char_model = CharMarkovChain(order=int(order))
    char_model.train(text)
    stats = char_model.stats()
    return (
        f"✅ Char-level model trained!\n\n"
        + "\n".join(f"  {k}: {v}" for k, v in stats.items())
    )


def generate_word(seed: str, max_words: int) -> str:
    try:
        return word_model.generate(seed=seed, max_words=int(max_words))
    except Exception as e:
        return f"❌ {e}"


def generate_char(seed: str, max_chars: int) -> str:
    try:
        return char_model.generate(seed=seed, max_chars=int(max_chars))
    except Exception as e:
        return f"❌ {e}"


def compare_orders(corpus_choice: str, seed: str) -> str:
    text = CORPORA.get(corpus_choice, list(CORPORA.values())[0])
    output = []
    for order in [1, 2, 3, 4]:
        m = WordMarkovChain(order=order)
        m.train(text)
        gen = m.generate(seed=seed, max_words=40)
        output.append(f"**Order {order}:**\n{gen}\n")
    return "\n".join(output)


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Task-03 · Markov Chain Text Generation",
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="teal",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
        .gradio-container { max-width: 1000px; margin: auto; }
        #title { text-align: center; padding: 20px 0 10px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# ⛓️ Task-03 — Text Generation with Markov Chains
**Statistical n-gram models for coherent text generation — no neural network required.**
            """,
            elem_id="title",
        )

        with gr.Tabs():
            # ── Word-level ────────────────────────────────────────────────
            with gr.TabItem("🔤 Word-Level"):
                with gr.Row():
                    with gr.Column():
                        corpus_dd = gr.Dropdown(
                            choices=list(CORPORA.keys()),
                            value="Tech Blog",
                            label="Choose Corpus",
                        )
                        custom_text = gr.Textbox(
                            label="Or paste custom training text",
                            lines=5,
                            placeholder="Paste any text here to train on it…",
                        )
                        order_sl = gr.Slider(1, 5, value=2, step=1, label="Markov Order (n)")
                        train_btn = gr.Button("🏋️ Train Word Model", variant="secondary")
                        train_status = gr.Textbox(label="Training Status", interactive=False, lines=6)

                    with gr.Column():
                        seed_in = gr.Textbox(label="Seed text (optional)", placeholder="the future of…")
                        words_sl = gr.Slider(20, 300, value=100, step=10, label="Max Words")
                        gen_btn = gr.Button("✍️ Generate", variant="primary")
                        output_box = gr.Textbox(label="Generated Text", lines=12, interactive=False)

                train_btn.click(fn=train_word_model, inputs=[corpus_dd, custom_text, order_sl], outputs=train_status)
                gen_btn.click(fn=generate_word, inputs=[seed_in, words_sl], outputs=output_box)

            # ── Char-level ────────────────────────────────────────────────
            with gr.TabItem("🔡 Character-Level"):
                with gr.Row():
                    with gr.Column():
                        corpus_dd2 = gr.Dropdown(
                            choices=list(CORPORA.keys()),
                            value="Shakespeare",
                            label="Choose Corpus",
                        )
                        custom_text2 = gr.Textbox(label="Or paste custom text", lines=5)
                        order_sl2 = gr.Slider(2, 8, value=4, step=1, label="Markov Order")
                        train_btn2 = gr.Button("🏋️ Train Char Model", variant="secondary")
                        train_status2 = gr.Textbox(label="Training Status", interactive=False, lines=5)

                    with gr.Column():
                        seed_in2 = gr.Textbox(label="Seed text (optional)", placeholder="to be…")
                        chars_sl = gr.Slider(100, 1000, value=400, step=50, label="Max Characters")
                        gen_btn2 = gr.Button("✍️ Generate", variant="primary")
                        output_box2 = gr.Textbox(label="Generated Text", lines=12, interactive=False)

                train_btn2.click(fn=train_char_model, inputs=[corpus_dd2, custom_text2, order_sl2], outputs=train_status2)
                gen_btn2.click(fn=generate_char, inputs=[seed_in2, chars_sl], outputs=output_box2)

            # ── Order comparison ─────────────────────────────────────────
            with gr.TabItem("📊 Compare Orders"):
                gr.Markdown("Compare text coherence across different Markov orders.")
                corpus_dd3 = gr.Dropdown(choices=list(CORPORA.keys()), value="Tech Blog", label="Corpus")
                seed_in3 = gr.Textbox(label="Seed", value="the future")
                compare_btn = gr.Button("🔬 Compare Orders 1–4", variant="primary")
                compare_out = gr.Markdown()
                compare_btn.click(fn=compare_orders, inputs=[corpus_dd3, seed_in3], outputs=compare_out)

            # ── About ─────────────────────────────────────────────────────
            with gr.TabItem("📖 About"):
                gr.Markdown(
                    """
## How Markov Chains Work

A Markov chain predicts the next token based solely on the previous **n** tokens (the order).

```
Order 1: P(w_t | w_{t-1})
Order 2: P(w_t | w_{t-2}, w_{t-1})
Order n: P(w_t | w_{t-n}, ..., w_{t-1})
```

### Training Phase
1. Slide a window of size **n** over the training text
2. Record which tokens follow each n-gram state
3. Build a frequency table → probability distribution

### Generation Phase
1. Start from a seed (or random state)
2. Sample the next token from the conditional distribution
3. Advance the window and repeat

### Order Trade-off
| Order | Coherence | Variety | Memorisation Risk |
|-------|-----------|---------|-------------------|
| 1 | Low | High | Low |
| 2 | Medium | Medium | Low |
| 3–4 | High | Lower | Medium |
| 5+ | Very High | Low | High |

### Stack
Pure Python · No ML dependencies · Gradio
                    """
                )

    return demo


if __name__ == "__main__":
    # Auto-train default models on startup
    word_model.train(CORPORA["Tech Blog"])
    char_model.train(CORPORA["Shakespeare"])
    print("[✓] Default models trained.")

    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7862)
