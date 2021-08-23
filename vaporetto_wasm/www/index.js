import init from '../pkg/vaporetto_wasm.js';
import * as wasm from '../pkg/vaporetto_wasm.js';

const loading = document.getElementById("loading");
loading.style.display = "block";

function run() {
    const predictor = wasm.Vaporetto.new();

    loading.style.display = "none";

    function createTextSpan(text) {
        const span = document.createElement("span");
        const textnode = document.createTextNode(text);
        span.appendChild(textnode);
        return span;
    }

    function replace_text(elem, prev_text, text, range_from, range_to, boundaries, window_size) {
        const prev_boundary_start = Math.max(range_from[0] - window_size, 0);
        const prev_boundary_end = Math.min(range_from[1] + window_size - 1, prev_text.length - 1);
        const node_end_idx = prev_boundary_end + 1;
        let node_end = elem.childNodes[0];
        if (prev_text.length != 0) {
            node_end = elem.childNodes[node_end_idx];
            if (range_from[0] == 0) {
                node_end.previousSibling.remove();
            }
            for (let i = prev_boundary_end - prev_boundary_start; i > 0; --i) {
                node_end.previousSibling.remove();
            }
        }
        const next_boundary_start = Math.max(range_to[0] - window_size, 0);
        const next_boundary_end = Math.min(range_to[1] + window_size - 1, text.length - 1);
        if (text.length != 0) {
            if (range_to[0] == 0) {
                node_end.before(createTextSpan(text[next_boundary_start]));
            }
            for (let i = 0; i < next_boundary_end - next_boundary_start; ++i) {
                const elem = createTextSpan(text[next_boundary_start + i + 1]);
                if (boundaries[i] >= 0) {
                    elem.style.borderLeft = '5pt solid rgba(0, 0, 0, ' + Math.atan(boundaries[i] / 2) + ')';
                }
                node_end.before(elem);
            }
        }
    }

    const input_text = document.getElementById('input_text');
    input_text.value = "";

    const window_size = 3;

    let input_data = null;
    let prev_range = [0, 0];
    let prev_chars = [];
    let chars_pos_map = [0];

    let composition_start = null;
    input_text.addEventListener('compositionstart', function (e) {
        composition_start = chars_pos_map[e.target.selectionStart];
    });

    input_text.addEventListener('compositionend', function (e) {
        composition_start = null;
    });

    input_text.addEventListener('beforeinput', function (e) {
        input_data = e.data;
        if (composition_start != null) {
            prev_range = [composition_start, chars_pos_map[e.target.selectionEnd]];
        } else {
            prev_range = [chars_pos_map[e.target.selectionStart], chars_pos_map[e.target.selectionEnd]];
        }
    });

    input_text.addEventListener('input', function (e) {
        const t0 = performance.now();

        const cur_text = e.target.value;
        const cur_chars = Array.from(cur_text);
        chars_pos_map = new Array(cur_text.length);
        let utf16_pos = 0;
        for (let i = 0; i < cur_chars.length; ++i) {
            chars_pos_map[utf16_pos] = i;
            utf16_pos += cur_chars[i].length;
        }
        chars_pos_map.push(cur_chars.length);

        let range_from = null;
        let range_to = null;
        switch (e.inputType) {
            case 'insertText':
            case 'insertLineBreak':
            case 'insertParagraph':
            case 'insertFromPaste':
            case 'insertCompositionText':
                range_from = prev_range;
                range_to = [prev_range[0], prev_range[1] + cur_chars.length - prev_chars.length];
                break;
            case 'deleteWordBackward':
            case 'deleteWordForward':
            case 'deleteSoftLineBackward':
            case 'deleteSoftLineForward':
            case 'deleteEntireSoftLine':
            case 'deleteHardLineBackward':
            case 'deleteHardLineForward':
            case 'deleteByCut':
            case 'deleteContent':
            case 'deleteContentBackward':
            case 'deleteContentForward':
                const start = chars_pos_map[e.target.selectionStart];
                const right_length = cur_chars.length - start;
                const prev_end = prev_chars.length - right_length;
                range_from = [start, prev_end];
                range_to = [start, start];
                break;
            default:
                range_from = [0, prev_chars.length];
                range_to = [0, cur_chars.length];
        }

        const tokenized = document.getElementById("tokenized");

        const predict_chars_start = Math.max(range_to[0] - window_size * 2 + 1, 0);
        const predict_chars_end = Math.min(range_to[1] + window_size * 2 - 1, cur_chars.length);
        const predict_chars = cur_chars.slice(predict_chars_start, predict_chars_end);

        const boundary_start = Math.max(range_to[0] - window_size, 0);
        const boundary_end = Math.min(range_to[1] + window_size - 1, cur_chars.length - 1);

        const predict_boundary_start = boundary_start - predict_chars_start;
        const predict_boundary_end = boundary_end - predict_chars_start;

        const boundaries = predictor.predict_partial(predict_chars.join(""), predict_boundary_start, predict_boundary_end);

        console.log("input with window:", predict_chars);
        console.log("prediction range:", [predict_boundary_start, predict_boundary_end]);
        console.log("boundaries:", boundaries);

        replace_text(tokenized, prev_chars, cur_chars, range_from, range_to, boundaries, window_size);

        const t1 = performance.now();

        console.log("Elapsed:", t1 - t0, "[ms]");
        console.log("-----");

        prev_chars = cur_chars;
    });
}

init().then(run);
