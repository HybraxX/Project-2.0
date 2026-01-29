/* =========================
   script.js — Full site logic (Updated with Gemini AI)
   ========================= */

/* -------------------------
   TRANSLATIONS / i18n
   ------------------------- */
console.log("script.js loaded");

const translations = {
  en: {
    welcome_message: 'Welcome back, Farmer!',
    location_title: 'Location Access Required',
    location_prompt: 'To provide localized weather and recommendations, Agri-Sight needs access to your location.',
    deny: 'Deny',
    allow: 'Allow',
    visual_diag_title: 'Visual Field Diagnostics',
    visual_diag_desc: 'Identify pests & diseases from a photo.',
    visual_diag_button: 'Start Scan',
    current_crop: 'Current Crop',
    sown_on: 'Sown on: 15 Nov 2024',
    predicted_yield: 'Predicted Yield',
    tons_per_hectare: 'Tons/Hectare',
    weather_forecast: 'Weather Forecast',
    weather_sunny: 'Sunny',
    live_monitoring: 'Live Field Monitoring (IoT Sensors)',
    soil_moisture: 'Soil Moisture',
    status_low: 'Low',
    last_update: 'Last update: just now',
    soil_temp: 'Soil Temperature',
    air_humidity: 'Air Humidity',
    soil_nutrients: 'Soil Nutrients (NPK)',
    nutrient_n: 'Nitrogen (N)',
    nutrient_p: 'Phosphorus (P)',
    status_good: 'Good',
    status_fair: 'Fair',
    recommendations: 'Actionable Recommendations',
    rec_irrigation_title: 'Immediate Action: Irrigation',
    rec_irrigation_body: 'Soil moisture is low (38%). Irrigate for 30 mins to reach optimal levels.',
    rec_pest_title: 'Pest Watch',
    rec_pest_body: 'High humidity detected. Increased risk of fungal growth. Scout fields.',
    yield_history: 'Yield History (Wheat)',
    crop_input_title: 'Crop Planner & Recommendations',
    crop_input_desc: "Enter the crop you'd like to plant and we'll recommend best fits based on your current soil.",
    weather_detail: 'Allow location to see local weather.'
  },
  hi: {
    welcome_message: 'वापस स्वागत है, किसान!',
    location_title: 'स्थान की अनुमति आवश्यक है',
    location_prompt: 'स्थानीय मौसम और सिफारिशें प्रदान करने के लिए, एग्री-साइट को आपके स्थान तक पहुंचने की आवश्यकता है।',
    deny: 'मना करें',
    allow: 'अनुमति दें',
    visual_diag_title: 'दृश्य क्षेत्र निदान',
    visual_diag_desc: 'एक तस्वीर से कीटों और बीमारियों को पहचानें।',
    visual_diag_button: 'स्कैन शुरू करें',
    current_crop: 'वर्तमान फसल',
    sown_on: 'बुवाई: 15 नवंबर 2024',
    predicted_yield: 'अनुमानित उपज',
    tons_per_hectare: 'टन/हेक्टेयर',
    weather_forecast: 'मौसम पूर्वानुमान',
    weather_sunny: 'धूप',
    live_monitoring: 'लाइव फील्ड मॉनिटरिंग (आईओटी सेंसर)',
    soil_moisture: 'मिट्टी की नमी',
    status_low: 'कम',
    last_update: 'अंतिम अपडेट: अभी',
    soil_temp: 'मिट्टी का तापमान',
    air_humidity: 'हवा में नमी',
    soil_nutrients: 'मिट्टी के पोषक तत्व (एनपीके)',
    nutrient_n: 'नाइट्रोजन (N)',
    nutrient_p: 'फॉस्फोरस (P)',
    status_good: 'अच्छा',
    status_fair: 'ठीक',
    recommendations: 'कार्रवाई योग्य सिफारिशें',
    rec_irrigation_title: 'तत्काल कार्रवाई: सिंचाई',
    rec_irrigation_body: 'मिट्टी की नमी कम (38%) है। इष्टतम स्तर तक पहुंचने के लिए 30 मिनट तक सिंचाई करें।',
    rec_pest_title: 'कीट निगरानी',
    rec_pest_body: 'अधिक नमी का पता चला। फंगल वृद्धि का खतरा बढ़ा। खेतों का निरीक्षण करें।',
    yield_history: 'उपज का इतिहास (गेहूँ)',
    crop_input_title: 'फसल योजनाकार और सिफारिशें',
    crop_input_desc: 'वह फसल दर्ज करें जिसे आप लगाना चाहते हैं और हम आपकी वर्तमान मिट्टी के अनुसार सर्वश्रेष्ठ विकल्प सुझाएंगे।',
    weather_detail: 'स्थानीय मौसम देखने के लिए स्थान सक्षम करें।'
  },
  bn: {
    welcome_message: 'ফিরে আসার জন্য স্বাগতম, কৃষক!',
    location_title: 'অবস্থান অ্যাক্সেস প্রয়োজন',
    location_prompt: 'স্থানীয় আবহাওয়া ও পরামর্শ প্রদানের জন্য, এগ্রি-সাইটকে আপনার অবস্থানে অ্যাক্সেস করতে হবে।',
    deny: 'অস্বীকার',
    allow: 'অনুমতি দিন',
    visual_diag_title: 'দৃশ্য ক্ষেত্র নির্ণয়',
    visual_diag_desc: 'ছবি থেকে কীটপতঙ্গ ও রোগ সনাক্ত করুন।',
    visual_diag_button: 'স্ক্যান শুরু করুন',
    current_crop: 'বর্তমান ফসল',
    sown_on: 'বপন: ১৫ নভেম্বর ২০২৪',
    predicted_yield: 'প্রত্যাশিত ফলন',
    tons_per_hectare: 'টন/হেক্টর',
    weather_forecast: 'আবহাওয়ার পূর্বাভাস',
    weather_sunny: 'রৌদ্রোজ্জ্বল',
    live_monitoring: 'লাইভ ফিল্ড মনিটরিং (আইওটি সেন্সর)',
    soil_moisture: 'মাটির আর্দ্রতা',
    status_low: 'কম',
    last_update: 'সর্বশেষ আপডেট: এখনই',
    soil_temp: 'মাটির তাপমাত্রা',
    air_humidity: 'বাতাসের আর্দ্রতা',
    soil_nutrients: 'মাটির পুষ্টি (এনপিকে)',
    nutrient_n: 'নাইট্রোজেন (N)',
    nutrient_p: 'ফসফরাস (P)',
    status_good: 'ভালো',
    status_fair: 'মধ্যম',
    recommendations: 'প্রযোজ্য পরামর্শ',
    rec_irrigation_title: 'তাত্ক্ষণিক পদক্ষেপ: সেচ',
    rec_irrigation_body: 'মাটির আর্দ্রতা কম (৩৮%)। সর্বোত্তম স্তরে পৌঁছাতে ৩০ মিনিট সেচ দিন।',
    rec_pest_title: 'কীটপতঙ্গ সতর্কতা',
    rec_pest_body: 'উচ্চ আর্দ্রতা সনাক্ত হয়েছে। ছত্রাক বৃদ্ধির ঝুঁকি বেশি। ক্ষেত পরীক্ষা করুন।',
    yield_history: 'ফলনের ইতিহাস (গম)',
    crop_input_title: 'ফসল পরিকল্পক ও পরামর্শ',
    crop_input_desc: 'যে ফসলটি আপনি লাগাতে চান তা লিখুন, এবং আমরা আপনার মাটির সাথে সবচেয়ে উপযুক্ত ফসল সাজেস্ট করব।',
    weather_detail: 'স্থানীয় আবহাওয়া দেখতে অবস্থান সক্ষম করুন।'
  }
};

function setLanguage(lang = 'en') {
  document.querySelectorAll('[data-lang-key]').forEach(el => {
    const key = el.getAttribute('data-lang-key');
    if (translations[lang] && translations[lang][key]) {
      if (el.tagName.toLowerCase() === 'input' || el.tagName.toLowerCase() === 'textarea') {
        el.value = translations[lang][key];
      } else {
        el.innerHTML = translations[lang][key];
      }
    }
  });

  // Number conversion helpers
  function toHindiDigits(str) {
    const map = { "0":"०","1":"१","2":"२","3":"३","4":"४","5":"५","6":"६","7":"७","8":"८","9":"९" };
    return str.toString().replace(/[0-9]/g, d => map[d]);
  }
  function toBengaliDigits(str) {
    const map = { "0":"০","1":"১","2":"২","3":"৩","4":"৪","5":"৫","6":"৬","7":"৭","8":"৮","9":"৯" };
    return str.toString().replace(/[0-9]/g, d => map[d]);
  }

  // Apply numeral conversion if Hindi or Bengali
  if (lang === 'hi' || lang === 'bn') {
    document.querySelectorAll("body *").forEach(el => {
      if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
        if (lang === 'hi') {
          el.textContent = toHindiDigits(el.textContent);
        } else if (lang === 'bn') {
          el.textContent = toBengaliDigits(el.textContent);
        }
      }
    });
  }
  document.documentElement.lang = lang;
}

/* wire language selector safely */
const langBtn = document.getElementById("langBtn");
const langMenu = document.getElementById("langMenu");
const currentLang = document.getElementById("currentLang");

if (langBtn) {
  langBtn.addEventListener("click", () => {
    langMenu.classList.toggle("show");
    langMenu.classList.toggle("hidden", !langMenu.classList.contains("show"));
  });

  langMenu.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      const lang = btn.dataset.lang;
      localStorage.setItem("lang", lang);
      currentLang.textContent = btn.textContent.trim();
      setLanguage(lang);
      langMenu.classList.remove("show");
      langMenu.classList.add("hidden");
    });
  });
}

/* -------------------------
   USER MENU
   ------------------------- */
const userMenuButton = document.getElementById('userMenuButton');
const userMenu = document.getElementById('userMenu');
if (userMenuButton && userMenu) {
  userMenuButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    userMenu.classList.toggle('show');
    userMenu.classList.toggle('hidden', !userMenu.classList.contains("show"));
  });
  document.addEventListener('click', (ev) => {
    if (!userMenu.contains(ev.target) && !userMenuButton.contains(ev.target)) {
      userMenu.classList.remove('show');
      userMenu.classList.add('hidden');
    }
  });
}

/* -------------------------
   LOCATION & WEATHER
   ------------------------- */
const locationModal = document.getElementById('locationModal');
const allowBtn = document.getElementById('allowLocation');
const denyBtn = document.getElementById('denyLocation');
const locationDisplay = document.getElementById('locationDisplay');
const weatherDisplay = document.getElementById('weatherDisplay');
const weatherDetail = document.getElementById('weatherDetail');
const weatherIcon = document.getElementById('weatherIcon');

// Replace with your OpenWeatherMap API key
const OPENWEATHER_API_KEY = 'd88e93af3e36ea377bf84d64f92ec221';

if (locationModal) {
  if (!navigator.geolocation) {
    locationModal.style.display = 'none';
    if (weatherDisplay) weatherDisplay.innerHTML = 'N/A';
    if (weatherDetail) weatherDetail.textContent = translations['en'].weather_detail;
  } else {
    locationModal.style.display = 'flex';
  }
}

function setWeatherUI({ temp, condition, description, city, country }) {
  if (weatherDisplay) weatherDisplay.innerHTML = `${Math.round(temp)}°C <span class="text-lg font-medium">${condition}</span>`;
  if (weatherDetail) weatherDetail.textContent = `Currently: ${description}.`;
  if (locationDisplay) {
    const span = locationDisplay.querySelector('span');
    if (span) span.textContent = `${city || ''}${country ? (city ? ', ' : '') + country : ''}`;
    locationDisplay.classList.remove('hidden');
    locationDisplay.classList.add('flex');
  }
  if (weatherIcon) {
    if (condition.includes('Cloud')) weatherIcon.className = 'fas fa-cloud text-gray-500 text-2xl';
    else if (condition.includes('Rain') || condition.includes('Drizzle')) weatherIcon.className = 'fas fa-cloud-showers-heavy text-blue-500 text-2xl';
    else if (condition.includes('Clear')) weatherIcon.className = 'fas fa-sun text-yellow-500 text-2xl';
    else weatherIcon.className = 'fas fa-smog text-gray-400 text-2xl';
  }
}

function fetchWeatherForCoords(lat, lon) {
  if (!OPENWEATHER_API_KEY) {
    if (weatherDisplay) weatherDisplay.innerHTML = '—';
    return;
  }
  const apiUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_API_KEY}&units=metric`;
  fetch(apiUrl)
    .then(r => { if (!r.ok) throw new Error('Weather fetch failed'); return r.json(); })
    .then(data => {
      const temp = data.main.temp;
      const condition = data.weather[0].main;
      const description = data.weather[0].description;
      const city = data.name;
      const country = data.sys && data.sys.country;
      setWeatherUI({ temp, condition, description, city, country });
    })
    .catch(err => {
      console.error('Weather fetch error', err);
      if (weatherDisplay) weatherDisplay.innerHTML = 'Error';
    });
}

if (allowBtn) {
  allowBtn.addEventListener('click', () => {
    if (locationModal) locationModal.style.display = 'none';
    if (!navigator.geolocation) return;
    
    if (weatherDisplay) weatherDisplay.innerHTML = 'Loading...';
    navigator.geolocation.getCurrentPosition(pos => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      fetchWeatherForCoords(lat, lon);
    }, err => {
      console.error('Geolocation error', err);
    });
  });
}
if (denyBtn) {
  denyBtn.addEventListener('click', () => {
    if (locationModal) locationModal.style.display = 'none';
    if (weatherDisplay) weatherDisplay.innerHTML = 'N/A';
  });
}

/* -------------------------
   IoT SENSORS
   ------------------------- */
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function safeSetWidth(id, width) {
  const el = document.getElementById(id);
  if (el && el.style) el.style.width = width;
}

function animateNumber(el, newValue, suffix = "") {
  if (!el) return;
  const current = parseFloat(el.textContent) || 0;
  const target = parseFloat(newValue);
  if (isNaN(target)) {
    el.textContent = newValue + suffix;
    return;
  }
  const duration = 800;
  const start = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const value = current + (target - current) * progress;
    el.textContent = value.toFixed(0) + suffix;
    if (progress < 1) requestAnimationFrame(step);
    else {
      el.classList.add("updated");
      setTimeout(() => el.classList.remove("updated"), 600);
    }
  }
  requestAnimationFrame(step);
}

// Global variable to store latest soil data for AI
window._LATEST_SOIL = { moisture: 38, ph: 6.4, temp: 22, nitrogen: 70, phosphorus: 55, potassium: 65, air_humidity: 62 };

function updateSensorUI(data) {
  if (!data) return;

  const m = data.soil_moisture ?? 38;
  const ph = (data.ph ?? 6.4);
  const ah = data.air_humidity ?? 62;
  const n = data.nitrogen ?? 70;
  const p = data.phosphorus ?? 55;
  const k = data.potassium ?? 65;
  const wl = data.water_level ?? 75;
  const at = data.air_temp ?? 29;
  const soilTemp = data.soil_temp ?? at;

  // 🌱 Animate numeric updates
  animateNumber(document.getElementById("soilMoistureValue"), m, "%");
  animateNumber(document.getElementById("airHumidityValue"), ah, "%");
  animateNumber(document.getElementById("soilTempValue"), soilTemp, "°C");
  animateNumber(document.getElementById("waterLevelValue"), wl, "%");

  // Static values
  safeSetText('soilPhValue', Number(ph).toFixed(1));

  // Nutrient bars
  safeSetWidth('nutNBar', `${n}%`);
  safeSetWidth('nutPBar', `${p}%`);
  safeSetWidth('nutKBar', `${k}%`);
  safeSetWidth('nutNBarSmall', `${n}%`);
  safeSetWidth('nutPBarSmall', `${p}%`);
  safeSetWidth('nutKBarSmall', `${k}%`);

  // Update nutrient status text
  const nStatus = n > 60 ? 'Good' : (n > 40 ? 'Fair' : 'Low');
  const pStatus = p > 60 ? 'Good' : (p > 40 ? 'Fair' : 'Low');
  const kStatus = k > 60 ? 'Good' : (k > 40 ? 'Fair' : 'Low');

  safeSetText('nutNStatusSmall', nStatus);
  safeSetText('nutPStatusSmall', pStatus);
  safeSetText('nutKStatusSmall', kStatus);

  // 🌊 Water Level Bars & Status
  const lowBar = document.getElementById("waterLowBar");
  const medBar = document.getElementById("waterMediumBar");
  const highBar = document.getElementById("waterHighBar");

  if (lowBar) lowBar.style.width = wl < 20 ? wl + "%" : "0%";
  if (medBar) medBar.style.width = (wl >= 20 && wl <= 80) ? wl + "%" : "0%";
  if (highBar) highBar.style.width = wl > 80 ? wl + "%" : "0%";

  const statusEl = document.getElementById("waterLevelStatus");
  if (statusEl) {
    if (wl > 80) {
      statusEl.textContent = "Status: High (Overflow Risk)";
      statusEl.className = "text-sm font-semibold mt-3 text-red-600";
    } else if (wl < 20) {
      statusEl.textContent = "Status: Low (Critical)";
      statusEl.className = "text-sm font-semibold mt-3 text-blue-600";
    } else {
      statusEl.textContent = "Status: Normal";
      statusEl.className = "text-sm font-semibold mt-3 text-green-600";
    }
  }

  // Save latest soil state for AI
  window._LATEST_SOIL = { moisture: m, ph: ph, temp: soilTemp, nitrogen: n, phosphorus: p, potassium: k, air_humidity: ah };
}

function fetchSensorData() {
  const mock = {
    soil_moisture: Math.floor(Math.random() * 40) + 30,
    ph: (Math.random() * 1.6 + 5.6).toFixed(1),
    air_humidity: Math.floor(Math.random() * 40) + 45,
    air_temp: Math.floor(Math.random() * 10) + 25,
    soil_temp: Math.floor(Math.random() * 6) + 20,
    water_level: Math.floor(Math.random() * 40) + 50,
    nitrogen: Math.floor(Math.random() * 40) + 50,
    phosphorus: Math.floor(Math.random() * 30) + 40,
    potassium: Math.floor(Math.random() * 30) + 45,
  };
  updateSensorUI(mock);
}
fetchSensorData();
setInterval(fetchSensorData, 12000);

/* -------------------------
   GEMINI AI: Soil Amendments
   ------------------------- */
async function suggestSoilAmendments() {
  const amendmentBox = document.getElementById("amendmentSuggestions");
  const suggestBtn = document.getElementById("suggestBtn");

  // 1. UI Loading State
  amendmentBox.innerHTML = `
    <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-3">
       <i class="fas fa-spinner fa-spin text-blue-600 text-xl"></i>
       <span class="text-blue-800 font-medium">Analyzing sensor data with AI...</span>
    </div>
  `;
  if (suggestBtn) suggestBtn.disabled = true;

  // 2. Gather Sensor Data from DOM
  const ph = document.getElementById("soilPhValue")?.innerText || "6.5";
  const moisture = document.getElementById("soilMoistureValue")?.innerText || "40%";
  const nStatus = document.getElementById("nutNStatusSmall")?.innerText || "Unknown";
  const pStatus = document.getElementById("nutPStatusSmall")?.innerText || "Unknown";
  const kStatus = document.getElementById("nutKStatusSmall")?.innerText || "Unknown";

  const payload = {
    ph: ph,
    moisture: moisture,
    n: nStatus,
    p: pStatus,
    k: kStatus
  };

  // 3. Call Backend
  try {
    const response = await fetch("http://127.0.0.1:5000/analyze-soil", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (data.error) {
       amendmentBox.innerHTML = `<p class="text-red-500 font-bold">Error: ${data.error}</p>`;
    } else {
       // Format text
       const formattedText = data.suggestion.replace(/\n/g, '<br>');
       amendmentBox.innerHTML = `
        <div class="p-5 bg-white border-l-4 border-blue-600 shadow-sm rounded-r-lg">
            <h4 class="font-bold text-gray-900 mb-2 flex items-center">
               <i class="fas fa-robot text-blue-600 mr-2"></i> AI Recommendation
            </h4>
            <div class="text-gray-700 text-sm leading-relaxed prose">
               ${formattedText}
            </div>
        </div>
       `;
    }
  } catch (err) {
    console.error(err);
    amendmentBox.innerHTML = `<p class="text-red-500">Failed to connect to the AI server. Is app.py running?</p>`;
  } finally {
    if (suggestBtn) suggestBtn.disabled = false;
  }
}

// Attach Event Listener
const suggestBtn = document.getElementById("suggestBtn");
if (suggestBtn) {
  suggestBtn.addEventListener("click", suggestSoilAmendments);
}

// Reset Button Logic
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    const amendmentBox = document.getElementById("amendmentSuggestions");
    if (amendmentBox) amendmentBox.innerHTML = "";
  });
}

/* -------------------------
   YIELD CHART (Chart.js)
   ------------------------- */
const yieldData = {
  Wheat: [3.5, 3.8, 3.6, 3.9, 3.7, 4.2],
  Rice:  [4.0, 4.2, 4.1, 4.3, 4.5, 4.8],
  Maize: [2.8, 3.0, 2.9, 3.2, 3.1, 3.5],
  Tomato:[20, 22, 21, 23, 24, 26],
  Potato:[18, 19, 18.5, 19.5, 20, 21]
};

let yieldChartInstance = null;

function initYieldChart(cropName = "Wheat") {
  const canvas = document.getElementById('yieldChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const yields = yieldData[cropName] || yieldData["Wheat"];
  const heading = document.getElementById("yieldChartHeading");
  if (heading) heading.textContent = `Yield History (${cropName})`;

  if (yieldChartInstance) yieldChartInstance.destroy();

  yieldChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024', '2025 (Pred.)'],
      datasets: [{
        label: `${cropName} Yield`,
        data: yields,
        backgroundColor: 'rgba(16,185,129,0.5)',
        borderColor: 'rgba(16,185,129,1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } }
    }
  });
}

/* -------------------------
   MARKET PRICE DATA & CHART
   ------------------------- */
const marketPriceData = {
  Wheat: [1850, 1920, 2000, 2100, 2050, 2200],
  Rice:  [1500, 1600, 1700, 1750, 1800, 1900],
  Maize: [1200, 1250, 1300, 1350, 1400, 1500],
  Tomato:[2200, 2500, 2400, 2600, 2700, 2800],
  Potato:[1000, 1100, 1050, 1150, 1200, 1250]
};

let marketChartInstance = null;

function initMarketChart(cropName = "Wheat") {
  const canvas = document.getElementById('marketChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const prices = marketPriceData[cropName] || marketPriceData["Wheat"];

  if (marketChartInstance) marketChartInstance.destroy();

  marketChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024', '2025 (Pred.)'],
      datasets: [{
        label: `${cropName} Price (₹/Quintal)`,
        data: prices,
        borderColor: 'blue',
        fill: false,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false
    }
  });
}

function updateChartsForActiveCrop() {
  const cropName = document.getElementById("currentCrop")?.textContent?.trim() || "Wheat";
  initYieldChart(cropName);
  initMarketChart(cropName);
}

/* -------------------------
   INIT & IMAGE UPLOAD
   ------------------------- */
window.addEventListener('load', () => {
  // Restore selected crop from URL or Storage
  const params = new URLSearchParams(window.location.search);
  const cropFromUrl = params.get('crop');
  const cropFromStorage = localStorage.getItem('selectedCrop');
  const chosen = cropFromUrl || cropFromStorage;

  if (chosen) {
     const currentCropEl = document.getElementById('currentCrop');
     if (currentCropEl) currentCropEl.textContent = chosen;
     localStorage.setItem('selectedCrop', chosen); // sync
  }

  updateChartsForActiveCrop();
  
  // Initialize weather
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(pos => {
      fetchWeatherForCoords(pos.coords.latitude, pos.coords.longitude);
    });
  }

  // Handle Image Upload Logic
  const imageInput = document.getElementById("cropImageInput");
  const previewContainer = document.getElementById("imagePreviewContainer");
  const previewImage = document.getElementById("imagePreview");
  const uploadStatus = document.getElementById("uploadStatus");
  const diseaseBox = document.getElementById("diseaseResult");

  if (imageInput) {
    imageInput.addEventListener("change", async function () {
      const file = this.files[0];
      if (!file) return;

      // Preview
      const reader = new FileReader();
      reader.onload = function (e) {
        if(previewImage) previewImage.src = e.target.result;
        if(previewContainer) previewContainer.classList.remove("hidden");
      };
      reader.readAsDataURL(file);

      // Status
      if (uploadStatus) {
        uploadStatus.textContent = "Uploading image...";
        uploadStatus.classList.remove("hidden");
      }

      const formData = new FormData();
      formData.append("image", file);

      try {
        await fetch("http://127.0.0.1:5000/upload-leaf", { method: "POST", body: formData });
        
        // Fetch prediction
        const predResp = await fetch("http://127.0.0.1:5000/predict-leaf");
        const predResult = await predResp.json();

        if (uploadStatus) uploadStatus.textContent = "Analysis Complete";
        
        if (diseaseBox) {
          diseaseBox.innerHTML = `Disease: <strong>${predResult.disease}</strong> (${predResult.confidence}%)`;
          diseaseBox.classList.remove("hidden");
        }
      } catch (err) {
        console.error(err);
        if (uploadStatus) uploadStatus.textContent = "Analysis failed. Check server.";
      }
    });
<<<<<<< HEAD
=======
    const uploadResult = await uploadResp.json();

    if (uploadStatus) {
      uploadStatus.textContent = uploadResult.message;
    }

    // 4️⃣ Fetch prediction
    const predResp = await fetch("http://127.0.0.1:5000/predict-leaf");
    const predResult = await predResp.json();

    console.log("Prediction:", predResult);
    
    if (diseaseBox) {
      diseaseBox.textContent =
        `Disease: ${predResult.disease} (${predResult.confidence}%)`;
      diseaseBox.classList.remove("hidden");
    }

  } catch (err) {
    console.error(err);
    if (uploadStatus) {
      uploadStatus.textContent = "Prediction failed";
    }
>>>>>>> 849fe05aa2bd0ab739617a39154d974fb2d1df48
  }
});