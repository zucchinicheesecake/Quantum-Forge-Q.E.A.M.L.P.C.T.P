export async function fetchChartData() {
  try {
    const response = await fetch('/api/chart-data');
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    return {
      labels: data.timestamps,
      values: data.prices
    };
  } catch (error) {
    throw new Error('Failed to fetch chart data: ' + error.message);
  }
}
