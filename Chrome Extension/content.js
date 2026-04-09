function cleanText(text) {
    return text
        .replace(/This transcript was generated automatically\. Its accuracy may vary\./gi, "")
        .replace(/\d+:\d+/g, "") // timestamps
        .replace(/\s+/g, " ")
        .trim();
}

function extractTranscript() {
    let transcript = [];

    const elements = Array.from(document.querySelectorAll('[data-encore-id="text"]'));
    let first = true
    // Spotify typische Textcontainer
    for (let el of elements) {
	if (el.tagName == 'SPAN' && el.className.includes('e-10180-text encore-text-body-small encore-internal-color-text-base')) {
	    if (first == true) {
	        first = false;
    	    } else {
                transcript.push(el.innerText)
	    }
        }
    }
    console.log(transcript);

    return transcript.join(" ");
}

function getTitle() {
    let title = document.querySelector('[data-testid="episodeTitle"]')?.innerText || "spotify_episode";
    let podcast = document.querySelector('[data-testid="showTitle"]')?.innerText || "unknown";

    return `${podcast} - ${title}`.replace(/[<>:"/\\|?*]/g, "");
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extract") {
	let text = extractTranscript();
        let title = getTitle();
	console.log(title + ": " + text);
        sendResponse({
            text: text,
            title: title
        });
    }
});