// ─── Char counter ───────────────────────────────────────────────────────────
const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");

newsText.addEventListener("input", () => {
    const len = newsText.value.length;
    charCount.textContent = `${len} character${len !== 1 ? "s" : ""}`;
});

// ─── Main analysis ──────────────────────────────────────────────────────────
async function checkNews() {

    const text = newsText.value.trim();

    if (!text) {
        showError("Please enter a news article or claim to analyze.");
        return;
    }

    // Reset UI to loading state
    setLoading(true);
    hideError();
    hideResult();

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResult(data);

    } catch (err) {
        if (err.name === "TypeError" && err.message.includes("fetch")) {
            showError("Cannot connect to server. Make sure the API is running on http://127.0.0.1:8000");
        } else {
            showError(err.message || "An unexpected error occurred.");
        }
    } finally {
        setLoading(false);
    }
}

// ─── Render result ───────────────────────────────────────────────────────────
function renderResult(data) {

    const {
        prediction,       // "Fake" | "Real"
        confidence,       // 0–1
        is_fake,          // boolean  ← was missing from original frontend
        emotional_intensity,
        suspicious_score,
        verification_score,
        truth_score
    } = data;

    const score = (truth_score * 100).toFixed(1);

    // ── Verdict banner ──────────────────────────────────────────────────────
    const banner = document.getElementById("verdictBanner");
    const icon   = document.getElementById("verdictIcon");
    const label  = document.getElementById("verdictLabel");
    const tag    = document.getElementById("isFakeTag");

    // Remove old verdict classes
    banner.classList.remove("verdict-real", "verdict-fake", "verdict-unsure");

    let verdictClass, iconChar, tagClass, tagText;

    if (is_fake) {
        verdictClass = "verdict-fake";
        iconChar     = "✕";
        tagClass     = "tag-fake";
        tagText      = "Flagged as Fake";
    } else if (truth_score >= 0.6) {
        verdictClass = "verdict-real";
        iconChar     = "✓";
        tagClass     = "tag-real";
        tagText      = "Likely Credible";
    } else {
        verdictClass = "verdict-unsure";
        iconChar     = "?";
        tagClass     = "";
        tagText      = "Low Confidence";
    }

    banner.classList.add(verdictClass);
    icon.textContent    = iconChar;
    label.textContent   = `Prediction: ${prediction}`;
    tag.textContent     = tagText;
    tag.className       = `tag ${tagClass}`;

    // ── Truth score bar ─────────────────────────────────────────────────────
    document.getElementById("truthScore").textContent = `${score}%`;

    const bar = document.getElementById("truthBar");
    bar.style.width = `${score}%`;
    bar.style.background =
        truth_score >= 0.7 ? "var(--green)" :
        truth_score >= 0.4 ? "var(--orange)" :
                             "var(--red)";

    // ── Metric cards ────────────────────────────────────────────────────────
    const confPct = (confidence * 100).toFixed(1);
    document.getElementById("confidence").textContent = `${confPct}%`;
    setBar("confidenceBar", confidence);

    const emotionAbs = Math.abs(emotional_intensity);
    document.getElementById("emotion").textContent = emotional_intensity.toFixed(3);
    setBar("emotionBar", emotionAbs);

    document.getElementById("suspicious").textContent = suspicious_score.toFixed(3);
    setBar("suspiciousBar", suspicious_score);

    document.getElementById("verification").textContent = verification_score.toFixed(3);
    setBar("verificationBar", verification_score);

    // Show result panel
    document.getElementById("resultBox").classList.remove("hidden");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function setBar(id, value) {
    document.getElementById(id).style.width = `${Math.min(value * 100, 100).toFixed(1)}%`;
}

function setLoading(active) {
    const btn      = document.getElementById("analyzeBtn");
    const label    = document.getElementById("btnLabel");
    const icon     = document.getElementById("btnIcon");
    const spinner  = document.getElementById("spinner");

    btn.disabled = active;
    label.textContent = active ? "Analyzing" : "Analyze";
    icon.classList.toggle("hidden", active);
    spinner.classList.toggle("hidden", !active);
}

function showError(msg) {
    const box = document.getElementById("errorBox");
    box.textContent = "⚠ " + msg;
    box.classList.remove("hidden");
}

function hideError() {
    document.getElementById("errorBox").classList.add("hidden");
}

function hideResult() {
    document.getElementById("resultBox").classList.add("hidden");
}