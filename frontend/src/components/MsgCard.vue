<template>
  <v-card w-auto :variant="msg.sender == 'user' ? 'tonal' : null" elevation="2"
    :color="msg.sender=='user' ? 'primary': null"
  >
    <v-card-text class="text-body-1" v-if="msg.text">
      {{ msg.text }}
    </v-card-text>
    <div class="temp" v-if="msg.entries">
      <v-list style="padding: 0px">
        <v-btn
          block
          variant="text"
          class="text-none"
          v-for="ent in msg.entries"
          @click.stop="sendEntry(ent)"
          :disabled="msg.disabled"
        >
          {{ ent }}
        </v-btn>
      </v-list>
    </div>

    <div class="temp" v-if="msg.options">
      <v-list lines="one">
        <v-checkbox
          v-model="selected"
          v-for="opt in msg.options"
          :label="opt"
          :value="opt"
          :disabled="msg.disabled"
          color="primary"
          hide-details
          h-30px
        >
        </v-checkbox>
      </v-list>
      <v-btn
        block
        class="bg-primary"
        variant="tonal"
        text="Continue"
        :disabled="msg.disabled"
        density="comfortable"
        @click.stop="sendSelected"
      >
      </v-btn>
    </div>

    <div class="temp" v-if="msg.table">
      <v-table density="compact">
        <thead>
          <tr>
            <th
              v-for="header in map(header_map, msg.table.header)"
              class="text-bold"
            >
              {{ header }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in msg.table.content">
            <td v-for="item in row">{{ item }}</td>
          </tr>
        </tbody>
      </v-table>
    </div>

    <div class="temp" v-if="msg.dropdown">
      <v-combobox
        label="Select food you want"
        :items="msg.dropdown"
        :disabled="msg.disabled"
        v-model="choosed"
        :rules="dropdownRules"
      >
      </v-combobox>
      <v-btn
        block
        class="bg-primary"
        variant="tonal"
        text="Confirm"
        :disabled="msg.disabled"
        density="comfortable"
        @click.stop="sendChoosed"
      >
      </v-btn>
    </div>

    <div class="temp" v-if="msg.plan">
      <v-timeline side="end" align="start">
        <v-timeline-item
          size="small"
          v-for="(day, i) in msg.plan"
          dot-color="primary"
        >
          <template v-slot:opposite>
            {{ "Day" + (i + 1) }}
          </template>

          <v-card elevation="0" v-for="meal in day.meals">
            <v-card-text class="text-subtitle-1 font-weight-medium pt-2 pb-1">{{
              meal.name + ":"
            }}</v-card-text>
            <v-card-text class="text-body-2 py-0"
              >Food items: {{ meal.items }}</v-card-text
            >
            <v-card-text class="text-body-2 py-0"
              >Calories: {{ meal.energy }}</v-card-text
            >
          </v-card>

          <v-card-text class="text-subtitle-2 font-weight-medium pb-0"
            >Daily consumption: {{ day.total[0] }} kcal.</v-card-text
          >
          <v-card-text class="text-subtitle-2 font-weight-medium py-0"
            >Total Daily Energy Expenditure: {{ day.total[1] }} kcal.</v-card-text
          >
          
        </v-timeline-item>
      </v-timeline>
    </div>

    <v-card-text class="text-body-1" v-if="msg.suffix">
      {{ map(suffix_map, msg.suffix) }}
    </v-card-text>

    <div class="temp" v-if="msg.confirm">
      <v-btn
        variant="tonal"
        density="comfortable"
        class="bg-success mx-1 mb-1"
        :text="msg.confirm[0]"
        @click.stop="sendEntry(msg.confirm[0])"
        :disabled="msg.disabled"
      >
      </v-btn>
      <v-btn
        variant="tonal"
        density="comfortable"
        class="bg-error mx-1 mb-1"
        :text="msg.confirm[1]"
        @click.stop="sendEntry(msg.confirm[1])"
        :disabled="msg.disabled"
      >
      </v-btn>
    </div>
  </v-card>
</template>

<script>
export default {
  props: ["msg"],

  data: () => ({
    suffix_map: {
      diagnosis:
        "The above diagnosis is based on the symptoms you have provided and is not a substitute for a medical diagnosis. Kindly consult a doctor for a proper diagnosis and treatment.",
    },
    header_map: {
      diagnosis: [
        "Disease",
        "Probability (%)",
        "Precaution 1",
        "Precaution 2",
        "Precaution 3",
        "Precaution 4",
      ],
      mealPlan: ["Food", "Energy (kcal)"],
      none: null,
    },

    choosed: undefined,
    selected: [],

    meals: ["Breakfast", "Lunch", "Dinner"],

    dropdownRules: [
      (value) => {
        if (value) return true;

        return "Must select one.";
      },
    ],
  }),

  methods: {
    map(obj, key) {
      if (key in obj) {
        return obj[key];
      } else {
        return key;
      }
    },

    sendSelected() {
      if (this.selected.length === 0) {
        this.$emit("sendMsg", "continue");
      } else {
        this.$emit("sendMsg", this.selected.join(","));
      }
    },

    sendEntry(value) {
      this.$emit("sendMsg", value);
    },

    sendChoosed() {
      if (this.choosed) {
        this.$emit("sendMsg", this.choosed);
      }
    },
  },
};
</script>

<style scoped>
div .temp {
  padding: 0.5%;
  margin-bottom: 0.5%;
}
</style>
