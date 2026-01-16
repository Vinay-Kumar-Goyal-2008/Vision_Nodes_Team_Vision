PROBLEM:
For over 43 million visually impaired individuals, web accessibility remains a critical barrier. Traditional screen readers struggle with complex, dynamic UIs, making navigation slow and linear. Meanwhile, voice-only tools are exhausting for fine control, and specialized hardware is often prohibitively expensive. Users are currently forced to choose between unreliable single-modal inputs, limiting their digital independence.

Solution: 
VisionNodes is a local-first, audio-visual OS that unifies gesture and voice control. It processes visual inputs (hand gestures) and audio commands in parallel to execute precise browser actions.It uses 8 agents differentiated by core responsibilities for automation like Wikipedia search summary , air quality , web summary , video summary , send emails , health chat , google books agent ( for education purpose to BLIND people ) , chat bot 

Core tech : google’s GestureRecognizer (CNN, pre-trained , static gestures) , google speech recognition , All-miniLM-L6-v2(sentence transformer, text to vector) , self pre-trained nanotemporal transformer using pytorch( dynamic gestures ) , playwright( web-automation ) , pyautoGUI ( windows-automation )  , OnDemand Agents  

IMPACT :
VisionNodes democratizes accessibility by transforming standard webcams and mic into powerful assistive tools, removing the need for expensive hardware. By combining the speed of gestures with the depth of voice AI, it turns the web from a slow, passive listening experience into a fast, interactive, and private environment for blind users.
