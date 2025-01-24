from PyQt5.QtChart import QChart
import gc

def purge_chart_memory(chart):
    """Nuclear-grade memory cleanup"""
    if not isinstance(chart, QChart):
        raise TypeError("Invalid chart object")
    chart.removeAllSeries()
    for axis in chart.axes():
        chart.removeAxis(axis)
        axis.deleteLater()
    if chart.scene():
        chart.scene().clear()
    chart.deleteLater()
    gc.collect()  # Force immediate cleanup
