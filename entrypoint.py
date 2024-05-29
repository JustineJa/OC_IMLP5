import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

# chargement du modèle
def pred_fn(text, pipeline, thresh, max_answers):
    pipe_output = pipeline(text, top_k=max_answers)
    recommended_tags = [
        dict_output['label'] for dict_output in pipe_output if dict_output['score'] > thresh
    ]
    
    return ", ".join(recommended_tags)


def main(text):
    device = 'cpu'
    model_path = 'JustineJ/OC_IMLP5'
    try: 
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\A475388\Notebooks\IML P5\Flask\bert_tokenizer')
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
    except:
        model = AutoModelForSequenceClassification.from_pretrained(
            'C:/Users/A475388/Notebooks/IML P5/Bert Antoine/bert_model_example'
        ) 
    model.to(device)


    pipe = TextClassificationPipeline(
        model=model, 
        tokenizer=tokenizer, 
        return_all_scores=False, # <-- renvoie les probas de tous les tags si True. 
                                # Sinon, renvoie uniquement le plus probable.
                                # ce paramètre sera ignoré si top_k est indiqué lors du call du pipe (voir plus bas)
        device=device,
        task="multi_label_classification",
        function_to_apply='sigmoid'
    )

    output = pred_fn(text, pipeline=pipe, thresh=0.5, max_answers=10)

    return output
    