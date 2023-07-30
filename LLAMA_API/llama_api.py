from flask import * 
import json, time
import torch
import transformers
from torch import cuda, bfloat16
import requests

model_id = 'meta-llama/Llama-2-13b-chat-hf'

#clinical self verification codes:

def get_extraction_prompt(note): 
    sys = "You are an expert in information extraction from clinical notes and patient doctor conversations."
    
    prompt = f"""
        Given a clinical note and patient doctor conversation you extract symptoms
        and return all as a single python list of strings like ["symptom","symptom","symptom"].
        Here is the note: {note} 
        
        Example output is here ###
        ["Abdominal pain", "pelvic pain", "swelling or lump", "abnormal bleeding", "urinary problems"]
    """
        #     Example output: 
        # ["Abdominal pain", "pelvic pain", "swelling or lump", "abnormal bleeding", "urinary problems"]
    return sys, prompt 

def get_evidence_prompt(note, symptoms): 
    sys = f"""
    You are an expert fact checker. Your job is to fact check the list of symptoms based on the note
    """ 
    # sys = """
    # Role: Medical expert fact checker. Check if symptoms are True or False using evidence from the full input note. State True or False and provide a one-sentence text span from the note as evidence.
    # """
    prompt = f"""
    Lets think logically: 
    step 1: For each symptom in the symptom list: {symptoms},  Determine if each symptom is correct or not based on the note 
    step 2: Find the span of text(one sentence exactly matched from note) that is evidence for each symptom in the note that makes the answer True or False
    step 3: return output as a python list of lists like [[symptom, evidence, True],[symptom, evidence, False]]
        
    Example output is here###
    
    [["Abdominal pain",  "I have pain near my stomach","True"],
     ["DM", " DM, on glipizide at home","True"],
     ["Hypertension", "high blood pressure ruled out", "False"]]
        
    Note is: {note}
    Output is: 
    """
    return sys, prompt
def get_evidence_prompt(note, symptoms): 
    sys = f"""
    You are an expert fact checker. Your job is to fact check the list of symptoms based on the note
    """ 
    # sys = """
    # Role: Medical expert fact checker. Check if symptoms are True or False using evidence from the full input note. State True or False and provide a one-sentence text span from the note as evidence.
    # """
    prompt = f"""
    Lets think logically: 
    step 1: For each symptom in the symptom list: {symptoms},  Determine if each symptom is correct or not based on the note 
    step 2: Find the span of text(one sentence exactly matched from note) that is evidence for each symptom in the note that makes the answer True or False
    step 3: return output as a python list of lists like [[symptom, evidence, True],[symptom, evidence, False]]
        
    Example output is here###
    
    [["Abdominal pain",  "I have pain near my stomach","True"],
     ["DM", " DM, on glipizide at home","True"],
     ["Hypertension", "high blood pressure ruled out", "False"]]
        
    Note is: {note}
    Output is: 
    """
    
def get_ultrasound_diagnosis_prompt(symptoms):
    sys = f"""You are an expert medical doctor, giving reliable diagnosis and advice"""
    prompt = f"""
        Given the following indications and evidences: {symptoms}, what kind of ultrasound should the doctor administer and use the evidences and show how confident you are that each evidence supports the conclusion
    """ 
    return sys, prompt


def extract_symptoms(text):
    symptoms = text.replace('"',"'")
    symptoms = symptoms.split("['")
    symptoms = [i for i in symptoms if "']" in i]
    symptoms = [i.split("']") for i in symptoms]
    symptoms = [i.split("', '") for i in symptoms[0] if i!=''][0]
    # print(str(symptoms))

    return str(symptoms)

def extract_evidence(text):
    symptoms = text.replace('"',"'")
    symptoms = symptoms.split("[['")
    symptoms = [i for i in symptoms if "']]" in i]
    symptoms = [i.split("']]") for i in symptoms]
    symptoms = "[['" + str([i for i in symptoms[0] if ('True' in i or 'False' in i)][0]) +"']]"

    return str(symptoms)

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
generate_text_demo = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home_page():
    data_set = {'Page': 'Home', 'Message': 'successsfully loaded the home page', 'Timestamp': time.time()}
    json_dump = json.dumps(data_set) 

    return json_dump

@app.route('/llama2/', methods = ['POST'])
def llama_api():
    data = request.json
    # Your data processing logic here...
    res = generate_text_demo(f"""{data["input"]}""")
    #res = generate_text(f"""<s>[INST] <<SYS>>
#You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

#If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
#<</SYS>>{data["input"]}[/INST]""")
    
    # Return the response in JSON format
    return jsonify({"message": "Data processed successfully", "result": res})


@app.route('/llama2_verify/',methods= ["POST"] )
def llama_verify():
    inputs = request.json
    prompt = f"""<s>[INST]<<SYS>>{inputs["SYS"]}<</SYS>>{inputs["INPUT"]}"""
    res = generate_text(prompt)
    return jsonify({"message": "Data processed successfully", "result":res})

@app.route('/llama2_evidence', methods = ["POST"])
def llama_evidence():
    note = request.json["input"]
    sys, prompt = get_extraction_prompt(note)
    response = make_request(prompt, sys, url)

    symptoms = extract_symptoms(response)
    print(symptoms)

    sys, prompt = get_evidence_prompt(note, symptoms)
    response = make_request(prompt, sys)
    print(response)


    symptoms_and_evidences = extract_evidence(response)

    sys, prompt = get_ultrasound_diagnosis_prompt(symptoms_and_evidences)
    response = make_request(prompt, sys)
    return jsonify({"message": "Data processed successfully", "symptoms":symptoms, "evidence":symptoms_and_evidences, "answer":response})

url = 'https://dynamic-condor-reliably.ngrok-free.app/llama2_verify/'  # Change this URL if needed


def make_request(prompt, sys=""):
    if not sys:
        print("no sys added, putting in blank string")
    input = f"""<s>[INST]<<SYS>>{sys}<</SYS>>{prompt}"""
    res = generate_text(prompt)

    return res



if __name__ == '__main__':
    app.run(port=7711)
