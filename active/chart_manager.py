class TestPurgeChartMemory:
	def test_valid_chart_cleanup(self):
		chart = QChart()
		# Arrange: Add some series and axes to the chart
		series = QLineSeries()
		chart.addSeries(series)
		chart.createDefaultAxes()
		
		# Act: Call the purge_chart_memory function
		purge_chart_memory(chart)
		
		# Assert: Check that the chart is cleaned up
		assert len(chart.series()) == 0
		assert len(chart.axes()) == 0