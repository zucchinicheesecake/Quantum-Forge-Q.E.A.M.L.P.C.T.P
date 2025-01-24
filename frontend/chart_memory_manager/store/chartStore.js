import { reactive } from 'vue';
import { fetchChartData } from '../api/chartApi';

export const chartStore = reactive({
  chartData: {
    labels: [],
    datasets: [
      {
        label: 'Price',
        data: [],
        borderColor: '#4CAF50',
        fill: false,
        pointRadius: 2,
        borderWidth: 2
      }
    ]
  },
  loading: false,
  error: null,
  
  async updateData() {
    this.loading = true;
    this.error = null;
    
    try {
      const response = await fetchChartData();
      this.chartData.labels = response.labels;
      this.chartData.datasets[0].data = response.values;
    } catch (error) {
      this.error = error.message;
      console.error('Error fetching chart data:', error);
    } finally {
      this.loading = false;
    }
  },
  
  clearData() {
    this.chartData.labels = [];
    this.chartData.datasets[0].data = [];
  }
});
