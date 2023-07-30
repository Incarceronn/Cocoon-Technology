# Cocoon Technologies - An Ultrasound Physician Assistant
*Transformative integration of state-of-the-art AI and portable ultrasound scanner*

# Key components

**DoctorAI**  
DoctorAI is an engine-api-telegram bot that contains over 1000 general cases from NHS.UK, along with LLM and vector DB. It operates under an open government license agreement to offer diagnoses and recommendations based on patient symptoms, with a particular focus on ultrasound usage for diagnosis.

**Ultrasound Training Assistant**  
Ultrasound Training Assistant is an engine-telegram bot that uses 246 existing training videos collected from Bufferfly Network Education. It provides training recommendations using LLM.

**Clinical verification**  
Clinical verification (Decide on choice of ultrasound method, backed up with evidence.)

**LLAMA 2**  
LLAMA 2 7B is the locally deployed core LLM engine, with OpenAI GPT3.5 as a backup, fine-tuned with 20 million tokens related to medical/patient conversations for medical purposes.
# Weights
Running the Llama 2 API will automatically download weights. To use pretrained weights, refer to code in last cell of Training.ipynb and swap in the model loading with in main2.py.

**LLAMA 2 API**
#run the app to load llama2 for inference 
python main2.py
#If you don't have an ngrok account, you will need to sign up at this link: https://ngrok.com/ 
#use ngrok to expose api to the web
ngrok http 7711 ...
#run req.py to send in a basic prompt to test if api is up 

# Train
**Hyperparameters**
LLAMA 2

Add data in csv format (follow format in mimic...,). Alternatively,  

Refer to Training.ipynb, Run all cells to train a finetuned llama. Adjust output_dir values to change where the model is saved. 

LLAVA
| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 128 | 2e-3 | 1 | 2048 | 0 |

# Inference
Restart the Training.ipynb notebook, restart the kernel run the first 2 and cells to test inference. To deploy on API, with a pretrained model, refer to LLAMA2 API section
# Evaluation
# Acknowledgement
LLAMA2: https://huggingface.co/blog/llama2 
clinical self verification: https://github.com/microsoft/clinical-self-verification.git


