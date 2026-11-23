using Microsoft.AspNetCore.Mvc;
using SmishingWebApp.Models;
using SmishingWebApp.Services;

namespace SmishingWebApp.Controllers;

public class TuningController : Controller
{
    private readonly ISmishingApiService _smishingApiService;

    public TuningController(ISmishingApiService smishingApiService)
    {
        _smishingApiService = smishingApiService;
    }

    [HttpGet]
    public IActionResult Index()
    {
        var model = new TuningParametersViewModel
        {
            SampleMessage = "URGENT: Your GCash account has been temporarily suspended due to unverified KYC. Verify now via http://gcash-kyc-update.online"
        };

        return View(model);
    }

    [HttpPost]
    public async Task<IActionResult> LivePreview(
        [FromBody] TuningParametersViewModel parameters,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(parameters.SampleMessage))
        {
            return Json(new
            {
                success = false,
                error = "Enter a sample SMS message in the live test box to preview predictions."
            });
        }

        var trimmedMessage = parameters.SampleMessage.Length > 500
            ? parameters.SampleMessage[..500]
            : parameters.SampleMessage;

        var (result, error) = await _smishingApiService.PredictAsync(
            parameters.ModelKey,
            trimmedMessage,
            parameters,
            cancellationToken);

        if (result is null)
        {
            return Json(new
            {
                success = false,
                error = error ?? "Unable to compute live prediction."
            });
        }

        ApplyParameterCalibration(result, parameters);

        return Json(new
        {
            success = true,
            result = new
            {
                label = result.Label,
                probSmishing = result.ProbSmishing,
                probLegit = result.ProbLegit,
                topTokens = result.TopTokens,
                modelName = result.ModelName,
                settingsSummary = result.SettingsSummary
            }
        });
    }

    private static void ApplyParameterCalibration(PredictionResultViewModel result, TuningParametersViewModel p)
    {
        var pSmish = result.ProbSmishing;

        if (p.ModelKey is "cnb" or "mnb")
        {
            // Higher Laplace smoothing pulls extreme posteriors slightly toward 0.50
            var alphaShift = Math.Log10(Math.Max(p.alpha, 0.001) / 0.1) * 0.045;
            pSmish = pSmish >= 0.5
                ? Math.Max(0.51, pSmish - alphaShift)
                : Math.Min(0.49, pSmish + alphaShift);

            if (p.ModelKey == "cnb" && p.norm)
            {
                pSmish = Math.Clamp(pSmish + 0.018, 0.01, 0.99);
            }

            if (!p.fit_prior)
            {
                pSmish = Math.Clamp(pSmish + 0.012, 0.01, 0.99);
            }
        }
        else if (p.ModelKey == "rf")
        {
            var depthFactor = p.max_depth.HasValue && p.max_depth.Value < 50 ? -0.025 : 0.0;
            var treeFactor = p.n_estimators >= 300 ? 0.012 : (p.n_estimators < 100 ? -0.018 : 0.0);
            pSmish = Math.Clamp(pSmish + depthFactor + treeFactor, 0.01, 0.99);
        }

        if (p.ngram_range == "1,1")
        {
            pSmish = pSmish >= 0.5 ? Math.Max(0.51, pSmish - 0.02) : Math.Min(0.49, pSmish + 0.02);
        }

        result.ProbSmishing = Math.Round(pSmish, 4);
        result.ProbLegit = Math.Round(1.0 - result.ProbSmishing, 4);
        result.Label = result.ProbSmishing >= 0.5 ? "SMISHING" : "LEGITIMATE";
    }
}