using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace SmishingWebApp.Models;

public class FastApiPredictRequest
{
    [JsonPropertyName("sms_message")]
    public string SmsMessage { get; set; } = string.Empty;
}

public class FastApiTokenDto
{
    [JsonPropertyName("token")]
    public string Token { get; set; } = string.Empty;

    [JsonPropertyName("weight")]
    public double Weight { get; set; }
}

public class FastApiPredictResponse
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("label")]
    public string Label { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("probabilities")]
    public Dictionary<string, double> Probabilities { get; set; } = new();

    [JsonPropertyName("top_tokens")]
    public List<FastApiTokenDto>? TopTokens { get; set; }
}

public class TokenSignal
{
    public string Token { get; set; } = string.Empty;
    public double Weight { get; set; }
}

public class PredictionResultViewModel
{
    public string Label { get; set; } = "LEGITIMATE";
    public double ProbSmishing { get; set; }
    public double ProbLegit { get; set; }
    public List<TokenSignal> TopTokens { get; set; } = new();
    public string ModelName { get; set; } = string.Empty;
    public string SettingsSummary { get; set; } = string.Empty;
}

public class TuningParametersViewModel
{
    public string ModelKey { get; set; } = "cnb";

    // Naive Bayes parameters
    [Range(0.001, 10.0)]
    public double alpha { get; set; } = 0.1;

    public bool fit_prior { get; set; } = true;

    public bool norm { get; set; } = false;

    // Random Forest parameters (matching bundled_pipeline.joblib)
    public string class_weight { get; set; } = "None";

    public string criterion { get; set; } = "entropy";

    [Range(1, 500)]
    public int? max_depth { get; set; } = 200;

    public string max_features { get; set; } = "log2";

    [Range(1, 50)]
    public int min_samples_leaf { get; set; } = 1;

    [Range(2, 50)]
    public int min_samples_split { get; set; } = 6;

    [Range(10, 1000)]
    public int n_estimators { get; set; } = 200;

    public int random_state { get; set; } = 42;

    // Shared TF-IDF text features
    public string ngram_range { get; set; } = "1,4";

    public bool lowercase { get; set; } = true;

    [Range(100, 50000)]
    public int max_vocab { get; set; } = 10000;

    public string? SampleMessage { get; set; }

    public PredictionResultViewModel? LiveResult { get; set; }
}

public class DetectorViewModel
{
    [Required(ErrorMessage = "Enter or extract an SMS message before running analysis.")]
    [StringLength(500, MinimumLength = 1, ErrorMessage = "Message length must be between 1 and 500 characters.")]
    public string SmsMessage { get; set; } = string.Empty;

    [Required]
    public string SelectedModel { get; set; } = "cnb";

    public TuningParametersViewModel Parameters { get; set; } = new();

    public PredictionResultViewModel? Result { get; set; }

    public string? ErrorMessage { get; set; }
}