import 'package:flutter/material.dart';
import '../../service/api_service.dart';
import '../../model/analysis_result.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService apiService = ApiService();
  final TextEditingController textController = TextEditingController();
  AnalysisResult? analysisResult;
  bool isLoading = false;

  Future<void> performAnalysis() async {
    setState(() {
      isLoading = true;
    });
    
    final result = await apiService.analyzeText(textController.text);
    
    setState(() {
      analysisResult = result;
      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Scam Detector')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: textController,
              decoration: InputDecoration(
                labelText: 'Enter text to analyze',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: isLoading ? null : performAnalysis,
              child: isLoading ? CircularProgressIndicator() : Text('Analyze'),
            ),
            SizedBox(height: 16),
            if (analysisResult != null)
              Expanded(
                child: ListView(
                  children: [
                    Text(
                      'Risk Score: ${analysisResult!.riskScore}',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    Text('Recommendation: ${analysisResult!.recommendation}'),
                    SizedBox(height: 8),
                    Text('Detected Patterns: ${analysisResult!.detectedPatterns.toString()}'),
                    SizedBox(height: 8),
                    Text('Sentiment Analysis: ${analysisResult!.sentimentAnalysis.toString()}'),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
