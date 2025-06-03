import os
import zipfile
import gdown
import torch
import requests
import json
import pandas as pd
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    MT5Tokenizer,
    MT5ForConditionalGeneration
)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
import zipfile
import shutil

# === дообученные модели ===

LOCAL_MODELS = {
    'mT5_m2o_russian_crossSum': '/content/gazeta_summarizer_mT5_m2o_russian_crossSum',
    'dorj': '/content/gazeta_summarizer_dorj',
    'sber_base': '/content/gazeta_summarizer_sber_base'
}

# === функция загрузки дообученных моделей ===
@st.cache_resource
def load_pretrained_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

MODELS = [
    {
        'label': 'utrobinmv/t5_summary_en_ru_zh_large_2048',
        'name': 'utrobinmv/t5_summary_en_ru_zh_large_2048',
        'prefix': 'summary: ',
        'tokenizer': T5Tokenizer,
        'model_class': T5ForConditionalGeneration
    },
    {
        'label': 'basil-77/rut5-base-absum-hh',
        'name': 'basil-77/rut5-base-absum-hh',
        'prefix': 'summary: ',
        'tokenizer': T5Tokenizer,
        'model_class': T5ForConditionalGeneration
    },
    {
        'label': 'cointegrated/rut5-base-absum',
        'name': 'cointegrated/rut5-base-absum',
        'prefix': 'summary: ',
        'tokenizer': T5Tokenizer,
        'model_class': T5ForConditionalGeneration
    },
    {
        'label': 'LOCAL: mT5_m2o_russian_crossSum (дообученная)',
        'name': LOCAL_MODELS['mT5_m2o_russian_crossSum'],
        'prefix': 'summarize: ',
        'tokenizer': AutoTokenizer,
        'model_class': AutoModelForSeq2SeqLM,
        'is_custom': True
    },
    {
        'label': 'csebuetnlp/mT5_multilingual_XLSum',
        'name': 'csebuetnlp/mT5_multilingual_XLSum',
        'prefix': 'summary: ',
        'tokenizer': MT5Tokenizer,
        'model_class': MT5ForConditionalGeneration,
        'temperature': 0.7
    },
    {
        'label': 'LOCAL: d0rj (дообученная)',
        'name': LOCAL_MODELS['dorj'],
        'prefix': 'summarize: ',
        'tokenizer': AutoTokenizer,
        'model_class': AutoModelForSeq2SeqLM,
        'is_custom': True
    },
    {
        'label': 'IlyaGusev/rut5_base_headline_gen_telegram',
        'name': 'IlyaGusev/rut5_base_headline_gen_telegram',
        'prefix': '',
        'tokenizer': T5Tokenizer,
        'model_class': T5ForConditionalGeneration
    },
    {
        'label': 'IlyaGusev/rut5_base_sum_gazeta',
        'name': 'IlyaGusev/rut5_base_sum_gazeta',
        'prefix': 'summarize: ',
        'tokenizer': T5Tokenizer,
        'model_class': T5ForConditionalGeneration,
        'repetition_penalty': 1.5
    },
    {
        'label': 'LOCAL: sber_base (дообученная)',
        'name': LOCAL_MODELS['sber_base'],
        'prefix': 'summarize: ',
        'tokenizer': AutoTokenizer,
        'model_class': AutoModelForSeq2SeqLM,
        'is_custom': True
    }
]





# === Настройки API для оценки ===
EVALUATION_MODELS = {
    # "Qwen": {
    #     "api_url": "https://api.intelligence.io.solutions/api/v1/chat/completions",
    #     "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    #     "api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjI4YjUzZmFiLWQ1YTItNGE2MC1hYWMyLTk2M2NjOGYwMTk1NCIsImV4cCI6NDkwMjI0MjcwNH0.cx6gxKFAsRA0BkCHkg_hHUsfHiYa98trzgUQ2QIuqB4Q3WHw8yQQCSeQdyCfq2Mtm3KpxHt2rE50OXu3tVzOsQ"
    # },
    "CohereForAI": {
        "api_url": "https://api.intelligence.io.solutions/api/v1/chat/completions",
        "model_name": "CohereForAI/aya-expanse-32b",
        "api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjI4YjUzZmFiLWQ1YTItNGE2MC1hYWMyLTk2M2NjOGYwMTk1NCIsImV4cCI6NDkwMjI0MjcwNH0.cx6gxKFAsRA0BkCHkg_hHUsfHiYa98trzgUQ2QIuqB4Q3WHw8yQQCSeQdyCfq2Mtm3KpxHt2rE50OXu3tVzOsQ"  # Замените на реальный ключ
    },
    "Mistral-Large": {
        "api_url": "https://api.intelligence.io.solutions/api/v1/chat/completions",
        "model_name": "mistralai/Mistral-Large-Instruct-2411",
        "api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjI4YjUzZmFiLWQ1YTItNGE2MC1hYWMyLTk2M2NjOGYwMTk1NCIsImV4cCI6NDkwMjI0MjcwNH0.cx6gxKFAsRA0BkCHkg_hHUsfHiYa98trzgUQ2QIuqB4Q3WHw8yQQCSeQdyCfq2Mtm3KpxHt2rE50OXu3tVzOsQ"  # Замените на реальный ключ
    },
    "Llama-3-70B": {
        "api_url": "https://api.intelligence.io.solutions/api/v1/chat/completions",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjI4YjUzZmFiLWQ1YTItNGE2MC1hYWMyLTk2M2NjOGYwMTk1NCIsImV4cCI6NDkwMjI0MjcwNH0.cx6gxKFAsRA0BkCHkg_hHUsfHiYa98trzgUQ2QIuqB4Q3WHw8yQQCSeQdyCfq2Mtm3KpxHt2rE50OXu3tVzOsQ"  # Замените на реальный ключ
    },
    "GLM-4": {
        "api_url": "https://api.intelligence.io.solutions/api/v1/chat/completions",
        "model_name": "THUDM/glm-4-9b-chat",
        "api_key": "io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6IjI4YjUzZmFiLWQ1YTItNGE2MC1hYWMyLTk2M2NjOGYwMTk1NCIsImV4cCI6NDkwMjI0MjcwNH0.cx6gxKFAsRA0BkCHkg_hHUsfHiYa98trzgUQ2QIuqB4Q3WHw8yQQCSeQdyCfq2Mtm3KpxHt2rE50OXu3tVzOsQ"  # Замените на реальный ключ
    }
}

# === Функции ===
@st.cache_resource
def load_model(model_name, tokenizer_class, model_class):
    """Загружает и кеширует модель суммаризации"""
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def parse_evaluation(evaluation_text):
    """Парсит текст оценки и извлекает числовые значения"""
    scores = {}
    lines = [line.strip() for line in evaluation_text.split('\n') if line.strip()]
    

    criterion_mapping = {
        'соответствие': 'Соответствие',
        'краткость': 'Краткость',
        'логичность': 'Логичность',
        'грамотность': 'Грамотность'
    }
    
    for line in lines:
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                criterion = parts[0].strip().lower()
                score = parts[1].strip()
                

                criterion = criterion_mapping.get(criterion, criterion)
                
                if score.isdigit():
                    scores[criterion] = int(score)
    return scores


# === ПРОМПТ ===
def evaluate_summary(original_text, summary, eval_model):
    """Оценивает summary через выбранную модель"""
    prompt = f"""
    Ты — эксперт в оценке качества реферативных summary (сжатых пересказов). 
    
    Инструкция:
    1) Оцени summary строго по 4 критериям (шкала 1-5).
    Критерии оценки:
    1. Соответствие (Factual Consistency):  
       - 5: Все ключевые факты сохранены, нет противоречий с оригиналом.  
       - 1: Есть серьёзные фактические ошибки или добавлена ложная информация.  
    
    2. Краткость (Compression Ratio):  
       - 5: Оптимальное сжатие (сохранено >80% смысла при <30% объёма).  
       - 1: Либо почти полный копипаст, либо потеря ключевой информации.  
    
    3. Логичность (Coherence & Structure):  
       - 5: Чёткая структура, причинно-следственные связи, плавные переходы.  
       - 1: Бессвязный набор фраз или нарушена последовательность.  
    
    4. Грамотность (Linguistic Quality):  
       - 5: Безупречный язык (орфография, пунктуация, стиль).  
       - 1: Много ошибок, мешающих пониманию.
       
    2) В ответе укажи ТОЛЬКО 4 цифры (по одной для каждого критерия). Правильный формат ответа СТРОГО должен выглядить так и никак иначе - навание критерия + оценка:  
       Соответствие: [оценка]
       Краткость: [оценка]
       Логичность: [оценка]
       Грамотность: [оценка]
    3) Никаких пояснений не добавляй.
    
    Дополнительные указания:
    1. Игнорируй субъективные предпочтения (например, "нравится/не нравится").  
    2. Жёстко penalize за галлюцинации (вымышленные факты).  
    
    Оригинал:  
    {original_text}  
    
    Summary для оценки:  
    {summary}  
    """

    data = {
        "model": eval_model["model_name"],
        "messages": [
            {"role": "system", "content": "Ты должен строго следовать формату: вывести только 4 оценки без комментариев."},
            {"role": "user", "content": prompt}
        ],
        # "temperature": 0.3,
        # "max_tokens": 200
        "temperature": 0.1,      
        "max_tokens": 300,       
        "stop": ["\n\n", "###"]       
    }

    headers = {
        "Authorization": f"Bearer {eval_model['api_key']}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            eval_model["api_url"],
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f" Ошибка оценки ({eval_model['model_name']}): {str(e)}"

# === Streamlit ===
st.set_page_config(page_title="Генерация и оценка Summary", layout="wide")
st.title(" Генерация Summary + Оценка")

text_input = st.text_area("Введите текст статьи:", height=300)

if st.button(" Сгенерировать и оценить"):
    if not text_input.strip():
        st.error("Введите текст!")
    else:
        results = []
        summary_ratings = {}
        
        for model_cfg in MODELS:
            with st.expander(f" {model_cfg['label']}", expanded=True):
                if model_cfg.get("is_custom") and not os.path.exists(model_cfg["name"]):
                    st.error(f"Модель не найдена по пути: {model_cfg['name']}")
                    continue
                
                with st.spinner("Генерация summary..."):
                    try:
                        if model_cfg.get("is_custom"):
                            tokenizer, model, device = load_pretrained_model(model_cfg["name"])
                        else:
                            tokenizer, model, device = load_model(
                                model_cfg['name'],
                                model_cfg['tokenizer'],
                                model_cfg['model_class']
                            )
                        
                        if tokenizer is None or model is None:
                            continue
                            
                        prefix = model_cfg['prefix'] or ""
                        input_text = prefix + text_input.strip()

                        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
                        output_ids = model.generate(**inputs, max_length=150)
                        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        
                        st.markdown("**Сгенерированное summary:**")
                        st.info(summary)
                        

                        evaluations = {}
                        for eval_name, eval_model in EVALUATION_MODELS.items():
                            with st.spinner(f"Оценка {eval_name}..."):
                                evaluation = evaluate_summary(text_input, summary, eval_model)
                                evaluations[eval_name] = evaluation
                                parsed_scores = parse_evaluation(evaluation)
                                
                                if parsed_scores:
                                    if model_cfg['label'] not in summary_ratings:
                                        summary_ratings[model_cfg['label']] = {}
                                    summary_ratings[model_cfg['label']][eval_name] = parsed_scores
                        
                        st.markdown("**Оценки:**")
                        for eval_name, evaluation in evaluations.items():
                            st.markdown(f"**{eval_name}:**")
                            st.code(evaluation)
                        
                        results.append({
                            'model': model_cfg['label'],
                            'summary': summary,
                            'evaluations': evaluations
                        })
                        
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
                        results.append({
                            'model': model_cfg['label'],
                            'error': str(e)
                        })
                
                st.divider()

        # таблица оценок
        if summary_ratings:
            st.subheader("📊 Итоговые средние оценки по моделям")
            

            criteria = ["Соответствие", "Краткость", "Логичность", "Грамотность"]
            model_names = list(summary_ratings.keys())
            
            df = pd.DataFrame(
                index=criteria,
                columns=model_names
            )
            
            for model in model_names:
                for criterion in criteria:
                    scores = []
                    for eval_model in EVALUATION_MODELS:
                        score = summary_ratings[model].get(eval_model, {}).get(criterion)
                        if score is not None:
                            scores.append(score)
                    
                    if scores:
                        df.loc[criterion, model] = sum(scores) / len(scores)
            

            df.loc['Общий балл'] = df.sum()
            

            st.dataframe(
                df.style
                .format("{:.2f}")
                .set_properties(**{'text-align': 'center'}),
                use_container_width=True
            )
            
            #  лучшая модель
            try:
                best_model = df.loc['Общий балл'].idxmax()
                best_score = df.loc['Общий балл'].max()
                st.success(f"🏆 **Лучшая модель:** {best_model} (сумма баллов: {best_score:.2f})")
            except Exception as e:
                st.warning(f"Не удалось определить лучшую модель: {e}")

        # Сохранение результатов
        try:
            with open("summary_evaluations.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            st.toast("Результаты успешно сохранены", icon="✅")
        except Exception as e:
            st.error(f"Ошибка сохранения результатов: {e}")