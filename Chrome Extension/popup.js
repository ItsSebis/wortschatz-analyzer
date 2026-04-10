document.getElementById("extract").addEventListener("click", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const response = await chrome.tabs.sendMessage(tab.id, {
        action: "extract"
    });

    const text = response.text;
    const filename = response.title + ".txt";
    console.log(text)
    // 👉 TEXT → FILE
    const blob = new Blob([text], { type: "text/plain" });
    const file = new File([blob], filename);

    let formData = new FormData();
    formData.append("file", file);

    try {
        await fetch("http://localhost:8081/uploadExtension", {
            method: "POST",
            body: formData
        });

        document.getElementById("status").innerText = "Uploaded";
    } catch (e) {
        document.getElementById("status").innerText = "Error";
    }
});