const SAMPLES = [
    "The petitioner was convicted of first-degree murder and sentenced to death. The defense argues that the confession was obtained through coercion and violated the defendant's Fifth Amendment rights against self-incrimination. The trial court failed to suppress the illegally obtained evidence.",
    "The employer discriminated against the plaintiff on the basis of race in violation of Title VII of the Civil Rights Act. The evidence shows a pattern of systematic exclusion of minority candidates from promotion opportunities within the organization.",
    "The state legislature enacted a statute prohibiting the publication of certain political advertisements within 30 days of a general election. The plaintiff newspaper argues this constitutes a prior restraint on speech in violation of the First Amendment."
];

document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.getElementById('inputText');
    const charCount = document.getElementById('charCount');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMsg = document.getElementById('errorMsg');
    const topPrediction = document.getElementById('topPrediction');
    const predictionsList = document.getElementById('predictionsList');

    // Update char count
    textarea.addEventListener('input', () => {
        charCount.textContent = `${textarea.value.length} chars`;
    });

    // Sample buttons
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = btn.getAttribute('data-sample');
            textarea.value = SAMPLES[idx];
            charCount.textContent = `${textarea.value.length} chars`;
            textarea.focus();
        });
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        textarea.value = '';
        charCount.textContent = '0 chars';
        results.classList.remove('active');
        errorMsg.classList.remove('active');
    });

    // Classify button
    classifyBtn.addEventListener('click', async () => {
        const text = textarea.value.trim();
        if (!text) {
            showError('Please enter some text to classify.');
            return;
        }

        classifyBtn.disabled = true;
        loading.classList.add('active');
        results.classList.remove('active');
        errorMsg.classList.remove('active');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
            } else {
                displayResults(data.results);
            }
        } catch (err) {
            showError('Server error. Make sure the Flask app is running.');
            console.error(err);
        } finally {
            classifyBtn.disabled = false;
            loading.classList.remove('active');
        }
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.add('active');
    }

    function displayResults(preds) {
        const top = preds[0];
        
        topPrediction.innerHTML = `
            <div style="font-size: 1.4rem; font-weight: 800; color: #818cf8; margin-bottom: 0.5rem;">
                ${top.label}
            </div>
            <div style="color: #94a3b8; font-size: 0.9rem;">
                Confidence: ${(top.score * 100).toFixed(1)}%
            </div>
        `;

        predictionsList.innerHTML = preds.map((p, i) => `
            <div class="prediction-item">
                <div style="width: 120px; font-weight: 600; font-size: 0.85rem;">${p.label}</div>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: 0%" data-width="${(p.score * 100).toFixed(1)}%"></div>
                </div>
                <div style="width: 50px; text-align: right; font-weight: 700; color: #818cf8; font-size: 0.85rem;">
                    ${(p.score * 100).toFixed(0)}%
                </div>
            </div>
        `).join('');

        results.classList.add('active');

        // Animate bars
        setTimeout(() => {
            document.querySelectorAll('.pred-bar').forEach(bar => {
                bar.style.width = bar.getAttribute('data-width');
            });
        }, 50);
    }
});
