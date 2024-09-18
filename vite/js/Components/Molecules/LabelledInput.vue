<script setup>
import Input from "@/Atoms/Input.vue";
import Label from "@/Atoms/Label.vue";

defineProps({
    disabled: {
        type: Boolean,
        default: false,
    },
    id: String,
    label: String,
    value: {
        type: [String, Number, Boolean, Object, Array],
        default: "",
    },
});
</script>

<template>
    <div
        class="mt-2 flex rounded-md shadow-sm">
        <Label
            :class="[disabled ? 'text-gray-400' : 'text-gray-500 dark:text-gray-200']"
            :for="id">
            <slot>{{ label }}</slot>
        </Label>
        <Input :id="id" :disabled="disabled" :value="value" />
    </div>
</template>

<style scoped>

</style>