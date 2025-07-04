import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq
from crossvit import CrossViT

# ─────────────────── Load DL model (CrossViT) ───────────────────
@st.cache_resource
def load_dl_model():
    model = CrossViT(image_size=224, channels=3, num_classes=2)
    state_dict = torch.load("C:/Users/HP ZBOOK FURY G7/Downloads/model_fold_5.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ─────────────────── Load Kosmos-2 LLM ───────────────────
@st.cache_resource
def load_llm():
    model_id = "microsoft/kosmos-2-patch14-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    return processor, model

# ─────────────────── Classify image with DL model ───────────────────
def classify_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    pred_class = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    return pred_class, confidence

# ─────────────────── Draw Entity Boxes ───────────────────
def draw_boxes(image, entities):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for _, _, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = [box[0]*w, box[1]*h, box[2]*w, box[3]*h]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image

# ─────────────────── Agentic Explanation via LLM ───────────────────
def agent_explanation(image, pred_class, confidence, processor, llm_model, user_prompt=""):
    label = {0: "No Fire", 1: "Fire", 2: "Smoke"}.get(pred_class, "Unknown")
    prompt = (
        f"<grounding>This image was classified as '{label}' with {confidence*100:.2f}% confidence.\n"
        f"What visual patterns might support this prediction (e.g., smoke, flames, surroundings)?"
    )
    if user_prompt.strip():
        prompt += f"\n\n<user_question>{user_prompt.strip()}</user_question>"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    output_ids = llm_model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        max_new_tokens=256,
    )
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    explanation, entities = processor.post_process_generation(decoded)
    return explanation, entities

# ───────────────────── Streamlit UI ─────────────────────
st.set_page_config(page_title="🔥 Agentic Fire Detector")
st.title("🧠 Agentic AI - Fire/Smoke Detection + Custom Prompt")

image_file = st.file_uploader("📷 Upload an RGB image", type=["jpg", "jpeg", "png"])
user_prompt = st.text_area("💬 Ask a follow-up question (optional)", placeholder="e.g., Is the smoke spreading?")

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running CrossViT classifier..."):
        dl_model = load_dl_model()
        pred_class, confidence = classify_image(image, dl_model)

    with st.spinner("🧠 Running Kosmos-2 Vision LLM..."):
        processor, llm_model = load_llm()
        explanation, entities = agent_explanation(image, pred_class, confidence, processor, llm_model, user_prompt)

    st.success("✅ Done!")
    st.markdown(f"### 📌 Prediction:\n**Class:** `{pred_class}` | **Confidence:** `{confidence*100:.2f}%`")
    st.markdown("### 🧠 LLM Explanation")
    st.info(explanation)

    if entities:
        st.markdown("### 🔍 Grounded Entities (Bounding Boxes)")
        st.image(draw_boxes(image.copy(), entities), caption="📌 Detected Regions", use_container_width=True)

        for text, _, boxes in entities:
            for box in boxes:
                st.markdown(f"- **{text}** → [x1={box[0]:.2f}, y1={box[1]:.2f}, x2={box[2]:.2f}, y2={box[3]:.2f}]")
