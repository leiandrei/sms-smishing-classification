using System.Net;
using System.Net.Http.Json;
using System.Text.RegularExpressions;
using SmishingWebApp.Models;

namespace SmishingWebApp.Services;

public interface ISmishingApiService
{
    Task<(PredictionResultViewModel? Result, string? Error)> PredictAsync(
        string modelKey,
        string smsMessage,
        TuningParametersViewModel? parameters = null,
        CancellationToken cancellationToken = default);
}

public partial class SmishingApiService : ISmishingApiService
{
    private static readonly Dictionary<string, string> ModelDisplayNames = new(StringComparer.OrdinalIgnoreCase)
    {
        ["cnb"] = "Complement Naive Bayes",
        ["mnb"] = "Multinomial Naive Bayes",
        ["rf"] = "Random Forest"
    };

    private static readonly Dictionary<string, double> HeuristicSignalWeights = new(StringComparer.OrdinalIgnoreCase)
    {
        ["verify"] = 0.89,
        ["urgent"] = 0.86,
        ["link"] = 0.84,
        ["http"] = 0.91,
        ["https"] = 0.88,
        ["gcash"] = 0.79,
        ["bdo"] = 0.78,
        ["otp"] = 0.83,
        ["suspended"] = 0.87,
        ["click"] = 0.85,
        ["claim"] = 0.82,
        ["prize"] = 0.81,
        ["php"] = 0.74,
        ["account"] = 0.72,
        ["expire"] = 0.80,
        ["kumpirmahin"] = 0.85,
        ["nanalo"] = 0.84
    };

    private readonly HttpClient _httpClient;

    public SmishingApiService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<(PredictionResultViewModel? Result, string? Error)> PredictAsync(
        string modelKey,
        string smsMessage,
        TuningParametersViewModel? parameters = null,
        CancellationToken cancellationToken = default)
    {
        var normalizedKey = ModelDisplayNames.ContainsKey(modelKey) ? modelKey.ToLowerInvariant() : "cnb";
        var payload = new FastApiPredictRequest { SmsMessage = smsMessage };

        try
        {
            using var response = await _httpClient.PostAsJsonAsync(
                $"/{normalizedKey}/predict",
                payload,
                cancellationToken);

            if (response.StatusCode == HttpStatusCode.ServiceUnavailable)
            {
                return (null, "Inference service unavailable (503). Verify that the FastAPI model bundle is loaded.");
            }

            if (response.StatusCode == HttpStatusCode.UnprocessableEntity)
            {
                return (null, "Invalid SMS message. Input must be between 1 and 500 characters.");
            }

            if (!response.IsSuccessStatusCode)
            {
                return (null, $"Prediction service returned HTTP {(int)response.StatusCode}.");
            }

            var dto = await response.Content.ReadFromJsonAsync<FastApiPredictResponse>(
                cancellationToken: cancellationToken);

            if (dto is null)
            {
                return (null, "Empty response received from inference service.");
            }

            var isSmishing = dto.Label.Contains("Smishing", StringComparison.OrdinalIgnoreCase)
                             || dto.Label.Contains("Scam", StringComparison.OrdinalIgnoreCase);

            var probSmishing = dto.Probabilities.FirstOrDefault(p =>
                p.Key.Contains("Smishing", StringComparison.OrdinalIgnoreCase) ||
                p.Key.Contains("Scam", StringComparison.OrdinalIgnoreCase)).Value;

            var probLegit = dto.Probabilities.FirstOrDefault(p =>
                p.Key.Contains("Legitimate", StringComparison.OrdinalIgnoreCase) ||
                p.Key.Contains("Ham", StringComparison.OrdinalIgnoreCase)).Value;

            if (probSmishing == 0 && probLegit == 0)
            {
                probSmishing = isSmishing ? dto.Confidence : 1.0 - dto.Confidence;
                probLegit = 1.0 - probSmishing;
            }

            var topTokens = dto.TopTokens is { Count: > 0 }
                ? dto.TopTokens.Select(t => new TokenSignal { Token = t.Token, Weight = t.Weight }).ToList()
                : ExtractFallbackTokens(smsMessage);

            var result = new PredictionResultViewModel
            {
                Label = isSmishing ? "SMISHING" : "LEGITIMATE",
                ProbSmishing = probSmishing,
                ProbLegit = probLegit,
                TopTokens = topTokens,
                ModelName = ModelDisplayNames[normalizedKey],
                SettingsSummary = BuildSettingsSummary(normalizedKey, parameters)
            };

            return (result, null);
        }
        catch (HttpRequestException)
        {
            return (null, "Cannot connect to the FastAPI backend on port 8000. Start the API container or Uvicorn server.");
        }
    }

    private static List<TokenSignal> ExtractFallbackTokens(string message)
    {
        var matches = WordRegex().Matches(message.ToLowerInvariant())
            .Select(m => m.Value)
            .Distinct();

        return matches
            .Where(token => HeuristicSignalWeights.ContainsKey(token) || token.Length >= 5)
            .Select(token => new TokenSignal
            {
                Token = token,
                Weight = HeuristicSignalWeights.TryGetValue(token, out var w) ? w : 0.42
            })
            .OrderByDescending(t => t.Weight)
            .Take(5)
            .ToList();
    }

    private static string BuildSettingsSummary(string modelKey, TuningParametersViewModel? p)
    {
        p ??= new TuningParametersViewModel { ModelKey = modelKey };

        return modelKey switch
        {
            "cnb" => $"alpha={p.alpha}, fit_prior={p.fit_prior.ToString().ToLowerInvariant()}, norm={p.norm.ToString().ToLowerInvariant()}, ngram_range=({p.ngram_range})",
            "mnb" => $"alpha={p.alpha}, fit_prior={p.fit_prior.ToString().ToLowerInvariant()}, ngram_range=({p.ngram_range})",
            "rf" => $"class_weight={p.class_weight}, criterion='{p.criterion}', max_depth={p.max_depth?.ToString() ?? "None"}, max_features='{p.max_features}', min_samples_leaf={p.min_samples_leaf}, min_samples_split={p.min_samples_split}, n_estimators={p.n_estimators}, random_state={p.random_state}",
            _ => $"ngram_range=({p.ngram_range})"
        };
    }

    [GeneratedRegex(@"[a-z0-9]{3,}")]
    private static partial Regex WordRegex();
}