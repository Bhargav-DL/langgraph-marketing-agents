import gradio as gr
from src.graph_builder import build_graph
from src.model_loader import load_model
from src.utils import create_call_qwen

# Load model and create call_qwen once
model, tokenizer = load_model()
call_qwen = create_call_qwen(model, tokenizer)

# Build the graph
app = build_graph(call_qwen)

def generate_content(topic, temperature):
    """Gradio interface function."""
    print("\n=== GRADIO FUNCTION CALLED ===")
    print("Topic:", topic)
    print("Temperature:", temperature)

    initial_state = {"topic": topic}
    final_state = app.invoke(initial_state)

    output = f"# 🚀 Marketing Content for: {topic}\n\n"
    output += "## 🔍 Web Research Findings\n"
    output += final_state.get('search_results', 'No results') + "\n\n"
    output += "## 📊 Market Research\n"
    output += final_state.get('research', '') + "\n\n"
    output += "## ✨ Final Content\n"
    output += final_state.get('final_content', '')
    return output

# Custom CSS for centered title
custom_css = """
    .gradio-container {
        text-align: center;
    }
    .gradio-container .title {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .gradio-container .subtitle {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1.5em;
    }
    .gradio-container .icon {
        display: none !important;
    }
"""

with gr.Blocks(title="AI Marketing Content Crew", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
        <div class="title">🎯 Multi-Agent Marketing Content Generator</div>
        <div class="subtitle">Powered by <strong>Qwen 2.5 7B</strong> + <strong>LangGraph</strong> + <strong>Web Research</strong><br>
        Your marketing team: <em>Search → Researcher → Strategist → Copywriter → SEO Editor</em></div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="Topic / Keyword",
                placeholder="e.g., 'AI for small business marketing'",
                lines=2
            )
        with gr.Column(scale=1):
            temp_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                label="Creativity"
            )

    submit_btn = gr.Button("✨ Generate Content", variant="primary")
    output_box = gr.Textbox(label="Generated Content", lines=20)

    submit_btn.click(
        fn=generate_content,
        inputs=[topic_input, temp_slider],
        outputs=output_box
    )

    gr.Examples(
        examples=[
            ["AI-powered personalization in e-commerce", 0.7],
            ["B2B SaaS content marketing trends 2026", 0.6],
            ["Sustainable packaging innovations", 0.8],
        ],
        inputs=[topic_input, temp_slider],
        outputs=output_box,
        fn=generate_content,
        cache_examples=False,
    )