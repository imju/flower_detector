
import gradio as gr
labels = learn_inf.dls.vocab
def predict(img):
    pred,pred_idx,probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Tulip/Rose/Daisy Flower Classifier"
description = "Tulip/Rose/Daisy flower classifier with fastai using Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://imju.me' target='_blank'>Blog post</a></p>"
interpretation='default'
enable_queue=True


gr.Interface(fn=predict, inputs=gr.Image(shape=(512, 512)), outputs=gr.Label(num_top_classes=3)).launch(share=True)