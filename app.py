"""Imports"""
from flask.templating import render_template_string
import requests
import torch
import string
import random
from fpdf import FPDF
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from flask import Flask,render_template,request

"""Initialising Flask"""
app = Flask(__name__,static_url_path='/static')
text = ""
output = ""
@app.route("/")
def landing_page():
    return render_template('index.html')

"""Summarising Function"""
@app.route("/summarise",methods = ["POST","GET"])
def index():
    print("file : "+str(len(request.form["input_text"]))+" Url: "+str(len(request.form["url"])))
    if not len(request.form["url"])==0:
        text=get_url_data(request.form["url"])
    elif not len(request.form["input_text"])==0:
        text=request.form["input_text"]
    else:
        return render_template('no_input.html')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
       
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device) 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=100,
                                        max_length=800,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print ("\n\nSummarized text: \n",output)    
    return render_template('result.html',var=text,var1=output)

"""Downloading result in a PDF format"""
@app.route("/download",methods = ["POST","GET"])
def download():
    in_text = request.form["input"]
    sum_text = request.form["output"]
    N=8
    res = ''.join(random.choices(string.ascii_uppercase ,k= N))
    filename="output_"+res+".txt"
    f = open(filename, "w",encoding="utf-8")
    f.write("Input Text \n")
    i=0
    for i in range(0,len(in_text),130):
        f.write(in_text[i:i+130]+"\n")
    f.write("Summarised Text\n")
    i=0
    for i in range(0,len(sum_text),130):
        f.write(sum_text[i:i+130]+"\n")
    return "Download successfull. Look for the following file in the directory for result "+filename

"""Getting content from Url"""
def get_url_data(url):
    req_obj = requests.get(url)
    text = req_obj.text
    soup = BeautifulSoup(text)
    all_paras = soup.find_all("p")
    url_text = ''
    for para in all_paras:
        url_text += para.text 
    print("Url Data: "+url_text)
    return url_text


"""Main function"""
if __name__ == "__main__":
    app.run(port = 9001,debug = True)