<template>
  <v-card
    w-auto
    :text="msg.text"
    :variant="msg.sender == 'user' ? 'tonal' : null"
    elevation="2"
  >
    <template v-if="msg.options">
      <v-list lines="one">
        <v-btn
          block
          variant="text"
          class="text-none text-left"
          v-for="(opt, i) in msg.options"
          :key="i"
          @click.stop="$emit('sendMsg',opt.value)"
        >
          {{ opt.text }}
        </v-btn>
      </v-list>
    </template>
  </v-card>
</template>

<script>
export default {
  props: ["msg"],
};
</script>