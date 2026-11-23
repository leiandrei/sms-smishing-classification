document.addEventListener("DOMContentLoaded", () => {
    const textarea = document.getElementById("smsMessageInput");
    const counter = document.getElementById("charCounter");
    const clearBtn = document.getElementById("clearMessageBtn");
    const form = document.getElementById("detectorForm");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const sampleButtons = document.querySelectorAll(".sample-pill");
    const dropzone = document.getElementById("ocrDropzone");
    const fileInput = document.getElementById("ocrFileInput");
    const ocrStatus = document.getElementById("ocrStatus");

    const updateCharCount = () => {
        if (!textarea || !counter) return;
        counter.textContent = `${textarea.value.length} / 500`;
    };

    if (textarea) {
        textarea.addEventListener("input", updateCharCount);
        updateCharCount();
    }

    sampleButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (!textarea) return;
            textarea.value = button.getAttribute("data-sample") || "";
            updateCharCount();
            textarea.focus();
        });
    });

    if (clearBtn && textarea) {
        clearBtn.addEventListener("click", () => {
            textarea.value = "";
            updateCharCount();
            textarea.focus();
        });
    }

    if (form && analyzeBtn) {
        form.addEventListener("submit", () => {
            if (textarea && textarea.value.trim().length > 0) {
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = "Analyzing...";
            }
        });
    }

    const runScreenshotOcr = async (file) => {
        if (!file || !file.type.startsWith("image/")) {
            if (ocrStatus) ocrStatus.textContent = "Please select a valid image file.";
            return;
        }

        if (typeof Tesseract === "undefined") {
            if (ocrStatus) ocrStatus.textContent = "OCR engine unavailable. Check internet connection for Tesseract.js.";
            return;
        }

        try {
            if (ocrStatus) ocrStatus.textContent = "Extracting Tagalog & English text on-device...";
            const result = await Tesseract.recognize(file, "eng+tgl", {
                logger: (info) => {
                    if (info.status === "recognizing text" && ocrStatus) {
                        const pct = Math.round((info.progress || 0) * 100);
                        ocrStatus.textContent = `Recognizing text on-device... ${pct}%`;
                    }
                }
            });

            const cleanedText = (result.data.text || "")
                .replace(/\s+/g, " ")
                .trim()
                .slice(0, 500);

            if (textarea) {
                textarea.value = cleanedText;
                updateCharCount();
                textarea.focus();
            }

            if (ocrStatus) {
                ocrStatus.textContent = cleanedText.length > 0
                    ? "Text extracted into message box. Review and click Analyze Message."
                    : "No readable text detected in screenshot.";
            }
        } catch {
            if (ocrStatus) ocrStatus.textContent = "Failed to extract text from screenshot.";
        }
    };

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                fileInput.click();
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files[0]) {
                runScreenshotOcr(fileInput.files[0]);
            }
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add("is-dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove("is-dragover");
            });
        });

        dropzone.addEventListener("drop", (event) => {
            const files = event.dataTransfer?.files;
            if (files && files[0]) {
                runScreenshotOcr(files[0]);
            }
        });
    }
});