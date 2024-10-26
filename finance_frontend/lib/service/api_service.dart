import 'dart:convert';
import 'package:http/http.dart' as http;
import '../model/analysis_result.dart';

class ApiService {
  static const String baseUrl = 'http://0.0.0.0:8000';

  Future<AnalysisResult?> analyzeText(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/analyze-text'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': text}),
    );

    if (response.statusCode == 200) {
      final data = json.decode(response.body)['analysis'];
      return AnalysisResult.fromJson(data);
    } else {
      print('Failed to fetch analysis: ${response.statusCode}');
      return null;
    }
  }
}
