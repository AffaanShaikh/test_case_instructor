document.getElementById('describeBtn').addEventListener('click', async () => {  
    const context = document.getElementById('context').value;  
    const imagesInput = document.getElementById('images');  
    const files = imagesInput.files;  
    if (files.length === 0) {  
     alert('Please upload at least one screenshot.');  
     return;  
    }  
    const imagesBase64 = await Promise.all([...files].map(file => {  
     return new Promise((resolve, reject) => {  
      const reader = new FileReader();  
      // get base64 string after the comma
      reader.onload = () => resolve(reader.result.split(',')[1]);  
      reader.onerror = reject;  
      reader.readAsDataURL(file);  
     });  
    }));  

    const outputDiv = document.getElementById('output'); 
    outputDiv.style.display = 'block'; // making the box visible after generation request
    outputDiv.innerHTML = '<p>Processing... Please wait.</p>';  
    try {  
     // fetch response
     const response = await fetch('/process', {  
      method: 'POST',  
      headers: {  
        'Content-Type': 'application/json',  
      },  
      body: JSON.stringify({  
        image: imagesBase64[0], // 
        text_input: context  
      })  
     });  
    
     const data = await response.json();  

     // clear previous response
     outputDiv.innerHTML = '';  

     displayOutput(data.output_text, outputDiv);  
    } catch (error) {  
     console.error('Error:', error);  
     outputDiv.innerHTML = '<p>An error occurred while processing your request. Please try again.</p>';  
    }  
  });  

  function displayOutput(outputTextArray, container) {  
    outputTextArray.forEach(outputText => {  
      // formatting for each test case
     const sections = outputText.split('\n\n');  
    
     sections.forEach(section => {  
      const sectionDiv = document.createElement('div');  
      const lines = section.split('\n');  
    
      lines.forEach(line => {  
        const paragraph = document.createElement('p');  
        //paragraph.innerText = line;  
        // replacing '**' with <strong> tags for bold text
        const formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        paragraph.innerHTML = formattedLine; 
        sectionDiv.appendChild(paragraph);
      });  
    
      container.appendChild(sectionDiv);  
     });  
    });  
  }