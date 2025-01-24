<template>
  <div class="chart-container">
    <canvas ref="chartCanvas"></canvas>
    <div class="chart-controls">
      <button @click="zoomIn">Zoom In</button>
      <button @click="zoomOut">Zoom Out</button>
      <button @click="resetZoom">Reset</button>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';

export default {
  props: ['chartData'],
  data() {
    return {
      chartInstance: null
    };
  },
  mounted() {
    this.renderChart();
  },
  methods: {
    renderChart() {
      Chart.register(zoomPlugin);
      
      this.chartInstance = new Chart(this.$refs.chartCanvas, {
        type: 'line',
        data: this.chartData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            zoom: {
              zoom: {
                wheel: {
                  enabled: true,
                },
                pinch: {
                  enabled: true
                },
                mode: 'xy',
              },
              pan: {
                enabled: true,
                mode: 'xy',
              }
            }
          },
          interaction: {
            mode: 'index',
            intersect: false,
          },
        }
      });
    },
    zoomIn() {
      if (this.chartInstance) {
        this.chartInstance.zoom(1.1);
      }
    },
    zoomOut() {
      if (this.chartInstance) {
        this.chartInstance.zoom(0.9);
      }
    },
    resetZoom() {
      if (this.chartInstance) {
        this.chartInstance.resetZoom();
      }
    }
  },
  watch: {
    chartData: {
      handler(newData) {
        if (this.chartInstance) {
          this.chartInstance.data = newData;
          this.chartInstance.update();
        }
      },
      deep: true
    }
  }
}
</script>

<style scoped>
.chart-container {
  position: relative;
  width: 100%;
  height: 500px;
}

.chart-controls {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 100;
}

.chart-controls button {
  margin: 2px;
  padding: 5px 10px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
}

.chart-controls button:hover {
  background-color: #45a049;
}
</style>
