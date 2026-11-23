document.addEventListener("DOMContentLoaded", () => {
    const modelSelect = document.getElementById("tuneModelKey");
    const nbFieldset = document.getElementById("nbFieldset");
    const rfFieldset = document.getElementById("rfFieldset");
    const normToggleWrapper = document.getElementById("normToggleWrapper");

    const alphaInput = document.getElementById("tuneAlpha");
    const alphaDisplay = document.getElementById("alphaValueDisplay");
    const fitPriorInput = document.getElementById("tuneFitPrior");
    const normInput = document.getElementById("tuneNorm");

    const estimatorsInput = document.getElementById("tuneEstimators");
    const estimatorsDisplay = document.getElementById("estimatorsValueDisplay");
    const maxDepthInput = document.getElementById("tuneMaxDepth");
    const maxDepthDisplay = document.getElementById("maxDepthValueDisplay");
    const minSamplesInput = document.getElementById("tuneMinSamplesSplit");
    const maxFeaturesSelect = document.getElementById("tuneMaxFeatures");

    const ngramSelect = document.getElementById("tuneNgramRange");
    const maxVocabInput = document.getElementById("tuneMaxVocab");
    const lowercaseInput = document.getElementById("tuneLowercase");

    const resetBtn = document.getElementById("resetDefaultsBtn");
    const useSettingsBtn = document.getElementById("useSettingsBtn");

    const sampleInput = document.getElementById("liveSampleInput");
    const syncStatus = document.getElementById("liveSyncStatus");
    const errorBox = document.getElementById("liveErrorBox");
    const resultContainer = document.getElementById("liveResultContainer");
    const verdictBadge = document.getElementById("liveVerdictBadge");
    const modelNameLabel = document.getElementById("liveModelName");
    const smishPctLabel = document.getElementById("liveSmishPct");
    const legitPctLabel = document.getElementById("liveLegitPct");
    const smishBar = document.getElementById("liveSmishBar");
    const legitBar = document.getElementById("liveLegitBar");
    const tokensList = document.getElementById("liveTokensList");
    const settingsSummary = document.getElementById("liveSettingsSummary");

    let debounceTimer = null;

    const getCurrentPayload = () => ({
        modelKey: modelSelect.value,
        alpha: parseFloat(alphaInput.value),
        fit_prior: fitPriorInput.checked,
        norm: normInput.checked,
        n_estimators: parseInt(estimatorsInput.value, 10),
        max_depth: parseInt(maxDepthInput.value, 10),
        min_samples_split: parseInt(minSamplesInput.value, 10),
        max_features: maxFeaturesSelect.value,
        ngram_range: ngramSelect.value,
        lowercase: lowercaseInput.checked,
        max_vocab: parseInt(maxVocabInput.value, 10),
        sampleMessage: sampleInput.value
    });

    const syncFieldsetStates = () => {
        const key = modelSelect.value;
        const isRf = key === "rf";
        const isCnb = key === "cnb";

        nbFieldset.disabled = isRf;
        nbFieldset.classList.toggle("is-disabled", isRf);

        rfFieldset.disabled = !isRf;
        rfFieldset.classList.toggle("is-disabled", !isRf);

        normInput.disabled = !isCnb;
        normToggleWrapper.classList.toggle("is-disabled", !isCnb);
    };

    const updateValueBadges = () => {
        alphaDisplay.textContent = parseFloat(alphaInput.value).toFixed(2);
        estimatorsDisplay.textContent = estimatorsInput.value;
        maxDepthDisplay.textContent = maxDepthInput.value;
    };

    const runLivePreview = async () => {
        const payload = getCurrentPayload();
        syncStatus.textContent = "UPDATING...";

        try {
            const response = await fetch("/Tuning/LivePreview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!data.success) {
                errorBox.textContent = data.error || "Live preview failed.";
                errorBox.classList.remove("hidden");
                resultContainer.classList.add("hidden");
                syncStatus.textContent = "ERROR";
                return;
            }

            errorBox.classList.add("hidden");
            resultContainer.classList.remove("hidden");

            const r = data.result;
            const isSmishing = r.label === "SMISHING";
            const smishPct = (r.probSmishing * 100).toFixed(2);
            const legitPct = (r.probLegit * 100).toFixed(2);

            verdictBadge.textContent = r.label;
            verdictBadge.className = `verdict-badge mono ${isSmishing ? "verdict-smishing" : "verdict-legit"}`;
            modelNameLabel.textContent = r.modelName;

            smishPctLabel.textContent = `${smishPct}%`;
            legitPctLabel.textContent = `${legitPct}%`;
            smishBar.style.width = `${smishPct}%`;
            legitBar.style.width = `${legitPct}%`;

            tokensList.innerHTML = "";
            (r.topTokens || []).forEach((item) => {
                const chip = document.createElement("span");
                chip.className = "token-chip mono";
                chip.innerHTML = `<span>${item.token}</span><span class="token-weight">${Number(item.weight).toFixed(2)}</span>`;
                tokensList.appendChild(chip);
            });

            settingsSummary.textContent = `Active: ${r.settingsSummary}`;
            syncStatus.textContent = "SYNCED";
        } catch {
            errorBox.textContent = "Unable to reach server for live preview.";
            errorBox.classList.remove("hidden");
            syncStatus.textContent = "OFFLINE";
        }
    };

    const schedulePreview = () => {
        updateValueBadges();
        syncFieldsetStates();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runLivePreview, 220);
    };

    [
        modelSelect,
        alphaInput,
        fitPriorInput,
        normInput,
        estimatorsInput,
        maxDepthInput,
        minSamplesInput,
        maxFeaturesSelect,
        ngramSelect,
        maxVocabInput,
        lowercaseInput,
        sampleInput
    ].forEach((el) => {
        if (!el) return;
        el.addEventListener("input", schedulePreview);
        el.addEventListener("change", schedulePreview);
    });

    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            modelSelect.value = "cnb";
            alphaInput.value = "0.10";
            fitPriorInput.checked = true;
            normInput.checked = false;
            estimatorsInput.value = "200";
            maxDepthInput.value = "200";
            minSamplesInput.value = "6";
            maxFeaturesSelect.value = "log2";
            ngramSelect.value = "1,4";
            maxVocabInput.value = "10000";
            lowercaseInput.checked = true;
            schedulePreview();
        });
    }

    if (useSettingsBtn) {
        useSettingsBtn.addEventListener("click", () => {
            const p = getCurrentPayload();
            const params = new URLSearchParams({
                modelKey: p.modelKey,
                alpha: String(p.alpha),
                fit_prior: String(p.fit_prior),
                norm: String(p.norm),
                n_estimators: String(p.n_estimators),
                max_depth: String(p.max_depth),
                min_samples_split: String(p.min_samples_split),
                max_features: p.max_features,
                ngram_range: p.ngram_range,
                lowercase: String(p.lowercase),
                max_vocab: String(p.max_vocab)
            });

            window.location.href = `/Home/Index?${params.toString()}`;
        });
    }

    updateValueBadges();
    syncFieldsetStates();
    runLivePreview();
});