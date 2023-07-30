import requests

url = 'https://dynamic-condor-reliably.ngrok-free.app/llama2'  # Change this URL if needed
query = """INDICATION:  ___ year-old female with cirrhosis, jaundice.

TECHNIQUE:"""  
query = """What do I do if I have blood in my stools?"""
context = """What is rectal bleeding?
Looking down into a toilet and seeing blood in your stool (poop) can be alarming. Your mind might go to many places as warning bells ring that something’s wrong. Rectal bleeding is a symptom of many different conditions, some more serious than others. It’s important to find out the cause of your rectal bleeding.

Some causes, like hemorrhoids, may not need treatment. But others, like colorectal cancer, need urgent care. Ulcers, anal fissures and inflammatory bowel disease (IBD) are other possible causes. A healthcare provider can help determine the cause of your hematochezia — the medical term for rectal bleeding or blood in your stool.

How does rectal bleeding appear?
You might see or experience rectal bleeding in a few different ways, including:

Noticing fresh blood on your toilet paper when you wipe.
Seeing blood in the bowl of the toilet after you use the bathroom. The water in the bowl might look like it’s been dyed red.
Seeing bright red, dark red or tarry black poop in the toilet.
When bleeding comes out from your anus (butthole), we call it rectal bleeding, but in fact, the bleeding could be coming from anywhere in your gastrointestinal (GI) tract. Your stomach, small intestine, colon, rectum and anus are all one continuous pathway, and all gastrointestinal bleeding comes out the same way.

What does blood in stool look like?
When you have blood in your stool, it can look a few different ways. You may have bright red streaks of blood on your poop, or you might see blood clots or blood and mucus mixed in with it. Your stool could also look dark, black and tarry. The color of the blood you see may be a clue to where it’s coming from:

Bright red blood in your stool usually means the bleeding is lower in your colon, rectum or anus.
Dark red or maroon blood can mean that you have bleeding higher up in your colon or your small intestine.
Melena (black stool) often points to bleeding in your stomach, such as a bleeding stomach ulcer.
Sometimes, rectal bleeding isn’t visible to the naked eye and can only be seen through a microscope. This is called occult bleeding. You may discover this type of blood in your stool if you have a lab test done on a stool sample called a fecal occult blood test. It’s a screening test for colorectal cancer.

Is blood in your stool serious?
Not necessarily, but it could be. It’s a good idea to check with a healthcare provider any time you have rectal bleeding or blood in your stool. Some minor conditions might not need treatment, but sometimes they might. Rectal bleeding could also be a sign of a more serious condition that needs treatment.

"""
data = {"input": query, "context":context}  # Your JSON data to send

response = requests.post(url, json=data)
print(response.json())
