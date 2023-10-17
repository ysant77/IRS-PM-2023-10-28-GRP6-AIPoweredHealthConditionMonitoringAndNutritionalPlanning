<template>
  <VSonner position="top-center"></VSonner>
  <v-card
    title="Edit your info and preferences:"
    width="700"
    class="mx-auto my-9 px-5 pb-10 pt-2"
  >
    <v-list-subheader>Basic info</v-list-subheader>
    <v-form @submit.prevent="submit">
      <v-text-field
        v-model="data.name"
        :rules="nameRules"
        label="Name"
      ></v-text-field>

      <v-select
        v-model="data.gender"
        :items="genders"
        :rules="genderRules"
        label="Gender"
      ></v-select>

      <v-text-field
        v-model="data.age"
        :rules="ageRules"
        label="Age"
      ></v-text-field>

      <v-text-field
        v-model="data.weight"
        :rules="wRules"
        label="Weight (kg)"
      ></v-text-field>
      <v-text-field
        v-model="data.height"
        :rules="hRules"
        label="Height (cm)"
      ></v-text-field>

      <v-divider></v-divider>
      <v-list-subheader>Food perferances</v-list-subheader>
      <v-checkbox
        hide-details
        color="info"
        label="Halal"
        v-model="data.is_halal"
      ></v-checkbox>
      <v-checkbox
        hide-details
        color="info"
        label="No beef"
        v-model="data.no_beef"
      ></v-checkbox>
      <v-checkbox
        hide-details
        color="info"
        label="Vegetarian"
        v-model="data.is_vegan"
      ></v-checkbox>

      <v-divider></v-divider>
      <v-list-subheader>Exercise amount (days/week)</v-list-subheader>
      <v-slider
        color="info"
        v-model="data.exec_lvl"
        :ticks="exec_labels"
        :max="5"
        :min="1"
        step="1"
        show-ticks="always"
        tick-size="4"
        class="pb-4"
      ></v-slider>

      <v-list-subheader>Weight goal</v-list-subheader>
      <v-slider
        color="info"
        v-model="data.weight_goal"
        :ticks="goal_labels"
        :max="3"
        :min="1"
        step="1"
        show-ticks="always"
        tick-size="4"
        class="pb-4"
      ></v-slider>

      <v-btn type="submit" block class="mt-2 bg-secondary large">confirm</v-btn>
    </v-form>
  </v-card>
</template>

<script>
import { hostname, axios } from "@/api/index";
import { alertToast } from "@/util.js";

import { VSonner } from "vuetify-sonner";

export default {
  components: {
    VSonner,
  },

  data: () => ({
    data: {
      uid: undefined,
      name: undefined,
      gender: undefined,
      age: undefined,
      weight: undefined,
      height: undefined,
      is_halal: false,
      no_beef: false,
      is_vegan: false,
      exec_lvl: 3,
      weight_goal: 2,
    },

    genders: ["male", "female"],
    exec_labels: {
      1: "Little",
      2: "1~3",
      3: "3~5",
      4: "6~7",
      5: "Heavy",
    },
    goal_labels: {
      1: "Lose weight",
      2: "Maintain weight",
      3: "Gain weight",
    },

    nameRules: [
      (value) => {
        if (value) return true;

        return "Required";
      },
    ],
    genderRules: [
      (value) => {
        if (value) return true;

        return "Required";
      },
    ],
    ageRules: [
      (value) => {
        if (value) return true;

        return "Required";
      },
      (value) => {
        if (/[0-9]/.test(value) && Number(value) > 0 && Number(value) <= 100)
          return true;

        return "Must be integer between 0 and 100";
      },
    ],
    hRules: [
      (value) => {
        if (value) return true;

        return "Required";
      },
      (value) => {
        if (
          /[0-9.]/.test(value) &&
          Number(value) >= 100 &&
          Number(value) <= 250
        )
          return true;

        return "Enter proper number 100~250.";
      },
    ],
    wRules: [
      (value) => {
        if (value) return true;

        return "Required";
      },
      (value) => {
        if (/[0-9.]/.test(value) && Number(value) > 0) return true;

        return "Enter proper number > 0.";
      },
    ],
  }),

  methods: {
    async submit(event) {
      let res = await event;
      if (res.valid) {
        await axios
          .post("http://" + hostname + "/api/curr-user-update", {
            data: JSON.stringify(this.data),
          })
          .then((res) => {
            if (res.data.status) {
              alertToast("Saved!", "success");
              setTimeout(() => {
                this.$router.back();
              }, 900);
            } else {
              alertToast("Failed!", "error");
            }
          })
          .catch((res) => {
            alertToast("Failed!", "error");
          });
      }
    },
  },

  mounted() {
    axios
      .get("http://" + hostname + "/api/curr-user")
      .then((res) => {
        this.data = res.data;
      })
      .catch((res) => {
        alertToast("Connection Failed!", "error");
      });
  },
};
</script>
