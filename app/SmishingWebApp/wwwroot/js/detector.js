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
    const ocrBadge = document.getElementById("ocrBadge");

    let ocrWorkerPromise = null;

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
            if (ocrStatus) {
                ocrStatus.textContent = "Drag and drop an SMS screenshot here, or click to select an image.";
            }
            if (ocrBadge) {
                ocrBadge.textContent = "ON-DEVICE";
            }
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

    const getOcrWorker = () => {
        if (ocrWorkerPromise) return ocrWorkerPromise;

        if (typeof Tesseract === "undefined") {
            return Promise.reject(new Error("Tesseract.js not loaded"));
        }

        ocrWorkerPromise = Tesseract.createWorker("eng", 1, {
            workerPath: "/lib/tesseract/worker.min.js",
            corePath: "/lib/tesseract/tesseract-core-simd-lstm.wasm.js",
            langPath: "/lib/tesseract",
            gzip: true
        }).catch((err) => {
            ocrWorkerPromise = null;
            throw err;
        });

        return ocrWorkerPromise;
    };

    const prepareImageForOcr = (file, maxWidth = 800) => {
        return new Promise((resolve) => {
            const img = new Image();
            const objectUrl = URL.createObjectURL(file);

            img.onload = () => {
                URL.revokeObjectURL(objectUrl);
                const scale = img.width > maxWidth ? maxWidth / img.width : 1;
                const canvas = document.createElement("canvas");
                canvas.width = Math.round(img.width * scale);
                canvas.height = Math.round(img.height * scale);

                const ctx = canvas.getContext("2d");
                if (!ctx) {
                    resolve(file);
                    return;
                }

                ctx.filter = "grayscale(100%) contrast(125%)";
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                canvas.toBlob((blob) => resolve(blob || file), "image/jpeg", 0.9);
            };

            img.onerror = () => {
                URL.revokeObjectURL(objectUrl);
                resolve(file);
            };

            img.src = objectUrl;
        });
    };

    const runScreenshotOcr = async (file) => {
        if (!file) return;

        const isImage = file.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(file.name);
        if (!isImage) {
            if (ocrStatus) ocrStatus.textContent = "Unsupported file type. Please choose a PNG, JPG, or WebP screenshot.";
            return;
        }

        if (ocrBadge) ocrBadge.textContent = "READING...";
        if (ocrStatus) ocrStatus.textContent = `Extracting text from "${file.name}"...`;

        try {
            const [worker, preparedImage] = await Promise.all([
                getOcrWorker(),
                prepareImageForOcr(file)
            ]);

            const output = await worker.recognize(preparedImage);
            const cleanedText = (output.data.text || "")
                .replace(/\s+/g, " ")
                .trim()
                .slice(0, 500);

            if (textarea) {
                textarea.value = cleanedText;
                updateCharCount();
                textarea.focus();
            }

            if (ocrBadge) ocrBadge.textContent = "DONE";
            if (ocrStatus) {
                ocrStatus.textContent = cleanedText.length > 0
                    ? `Extracted text from "${file.name}". Review and click Analyze Message.`
                    : `No readable text detected in "${file.name}".`;
            }
        } catch (err) {
            console.error("OCR extraction error:", err);
            if (ocrBadge) ocrBadge.textContent = "ERROR";
            if (ocrStatus) ocrStatus.textContent = "Could not extract text from screenshot. Check network access and try again.";
        } finally {
            if (fileInput) fileInput.value = "";
        }
    };

    if (dropzone && fileInput) {
        dropzone.addEventListener("pointerenter", () => {
            getOcrWorker().catch(() => {});
        }, { once: true });

        dropzone.addEventListener("click", () => {
            getOcrWorker().catch(() => {});
            fileInput.click();
        });

        dropzone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                getOcrWorker().catch(() => {});
                fileInput.click();
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
                runScreenshotOcr(fileInput.files[0]);
            }
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropzone.classList.add("is-dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropzone.classList.remove("is-dragover");
            });
        });

        dropzone.addEventListener("drop", (event) => {
            const files = event.dataTransfer?.files;
            if (files && files.length > 0) {
                runScreenshotOcr(files[0]);
            }
        });
    }
});