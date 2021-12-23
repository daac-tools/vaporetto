function createTextSpan(text, isBoundary, score) {
    const span = document.createElement("span");
    const textnode = document.createTextNode(text);
    span.appendChild(textnode);
    if (isBoundary) {
        span.style.borderLeft = "5pt solid rgba(0, 0, 0, " + Math.atan(score / 2) + ")";
    }
    return span;
}

vaporetto_bccwj_suw_small().then((Vaporetto) => {
    const vaporetto_suw = Vaporetto.new("DG");

    input_text.addEventListener("input", (e) => {
        const text = input_text.value;
        const scores = vaporetto_suw.predict_with_score(text);
        let i = -1;
        while (tokenized.firstChild) {
            tokenized.removeChild(tokenized.firstChild);
        }
        for (let c of text) {
            if (i >= 0) {
                tokenized.appendChild(createTextSpan(c, scores[i][0], scores[i][1] / 10000));
            } else {
                tokenized.appendChild(createTextSpan(c, false, 0));
            }
            ++i;
        }
    });
});
