class AnalysisResult {
  final double riskScore;
  final String recommendation;
  final Map<String, dynamic> detectedPatterns;
  final Map<String, dynamic> sentimentAnalysis;

  AnalysisResult({
    required this.riskScore,
    required this.recommendation,
    required this.detectedPatterns,
    required this.sentimentAnalysis,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      riskScore: json['risk_score'] ?? 0.0,
      recommendation: json['recommendation'] ?? '',
      detectedPatterns: json['detected_patterns'] ?? {},
      sentimentAnalysis: json['sentiment_analysis'] ?? {},
    );
  }
}
