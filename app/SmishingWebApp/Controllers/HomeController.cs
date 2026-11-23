using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using SmishingWebApp.Models;
using SmishingWebApp.Services;

namespace SmishingWebApp.Controllers;

public class HomeController : Controller
{
    private readonly ISmishingApiService _smishingApiService;

    public HomeController(ISmishingApiService smishingApiService)
    {
        _smishingApiService = smishingApiService;
    }

    [HttpGet]
    public IActionResult Index(
        string? modelKey,
        double? alpha,
        bool? fit_prior,
        bool? norm,
        int? n_estimators,
        int? max_depth,
        int? min_samples_split,
        string? max_features,
        string? ngram_range,
        bool? lowercase,
        int? max_vocab)
    {
        var selectedModel = string.IsNullOrWhiteSpace(modelKey) ? "cnb" : modelKey.ToLowerInvariant();

        var parameters = new TuningParametersViewModel
        {
            ModelKey = selectedModel,
            alpha = alpha ?? 0.1,
            fit_prior = fit_prior ?? true,
            norm = norm ?? false,
            n_estimators = n_estimators ?? 200,
            max_depth = max_depth ?? 200,
            min_samples_split = min_samples_split ?? 6,
            max_features = string.IsNullOrWhiteSpace(max_features) ? "log2" : max_features,
            ngram_range = string.IsNullOrWhiteSpace(ngram_range) ? "1,4" : ngram_range,
            lowercase = lowercase ?? true,
            max_vocab = max_vocab ?? 10000
        };

        var viewModel = new DetectorViewModel
        {
            SelectedModel = selectedModel,
            Parameters = parameters
        };

        return View(viewModel);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Index(DetectorViewModel model, CancellationToken cancellationToken)
    {
        model.Parameters.ModelKey = model.SelectedModel;

        if (!ModelState.IsValid)
        {
            return View(model);
        }

        var (result, error) = await _smishingApiService.PredictAsync(
            model.SelectedModel,
            model.SmsMessage,
            model.Parameters,
            cancellationToken);

        model.Result = result;
        model.ErrorMessage = error;

        return View(model);
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}