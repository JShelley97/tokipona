import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load model and tokenizer only once when the app starts
@st.cache_resource()
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona")
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    return model, tokenizer

model, tokenizer = load_model()

def translate(text, src_lang, tgt_lang):
    src = LANG_CODES.get(src_lang)
    tgt = LANG_CODES.get(tgt_lang)

    tokenizer.src_lang = src
    tokenizer.tgt_lang = tgt

    ins = tokenizer(text, return_tensors="pt").to("cpu")

    gen_args = {
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
        "length_penalty": 0.0,  # don't encourage longer or shorter output,
        "num_return_sequences": 3,
        "num_beams": 3,
        "forced_bos_token_id": tokenizer.lang_code_to_id[tgt],
    }

    outs = model.generate(**{**ins, **gen_args})
    output = tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)
    return output

st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Toki_pona.svg/800px-Toki_pona.svg.png',width=100)
st.title("Autismus Totalus")
st.write("aplicacion para ayudar a dos autistas con sus conversiones en Toki Pona")

LANG_CODES = {"English": "en", "toki pona": "tl"}


if 'src' not in st.session_state:
    st.session_state.src = 'English'
if 'trg' not in st.session_state:
    st.session_state.trg="toki pona"
if 'a' not in st.session_state:
    st.session_state.trg="toki pona"

output=""
if st.button("<>"):
    st.session_state.a=st.session_state.src
    st.session_state.src=st.session_state.trg
    st.session_state.trg=st.session_state.a
col1,col2=st.columns(2)
with col1: 
    st.write(st.session_state.src)
with col2:
    st.write(st.session_state.trg)
    

txt = st.text_input("texto para traducir")

if st.button("translate"):
    output = translate(txt, st.session_state.src, st.session_state.trg)
    st.write("1: ",output[0])
    st.write("2: ",output[1])
    st.write("3: ",output[2])
    





