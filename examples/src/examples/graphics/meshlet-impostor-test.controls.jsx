import {
    BindingTwoWay,
    BooleanInput,
    Label,
    LabelGroup,
    Panel,
    SelectInput,
    SliderInput
} from '@playcanvas/pcui/react';

/**
 * @import { Observer } from '@playcanvas/observer'
 * @import { ReactElement } from 'react'
 */

/**
 * @param {object} props - The props.
 * @param {Observer} props.observer - The observer.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    const slider = (/** @type {string} */ text, /** @type {string} */ path, /** @type {number} */ min, /** @type {number} */ max, precision = 2) => (
        <LabelGroup text={text}>
            <SliderInput
                binding={new BindingTwoWay()}
                link={{ observer, path: `data.${path}` }}
                min={min}
                max={max}
                precision={precision}
            />
        </LabelGroup>
    );
    return (
        <>
            <Panel headerText='Trees'>
                {slider('Prototype', 'prototype', 0, 15, 0)}
                <LabelGroup text='Showing'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.stats' }}
                        value={observer.get('data.stats')}
                    />
                </LabelGroup>
                <LabelGroup text='Force billboards'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.forceBillboards' }}
                    />
                </LabelGroup>
                {slider('Impostor px', 'impostorPx', 0, 600, 0)}
                <LabelGroup text='Impostor view'>
                    <SelectInput
                        type='number'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.debug' }}
                        options={[
                            { v: 0, t: 'Lit' },
                            { v: 1, t: 'Tint' },
                            { v: 2, t: 'Albedo' },
                            { v: 3, t: 'Normal' },
                            { v: 4, t: 'Alpha' }
                        ]}
                    />
                </LabelGroup>
            </Panel>
            <Panel headerText='Impostor shading'>
                {slider('Crown normal', 'bent', 0, 1)}
                {slider('Lower crown AO', 'occlusion', 0, 1)}
                {slider('Albedo scale', 'albedo', 0.2, 1.5)}
            </Panel>
            <Panel headerText='Light'>
                {slider('Sun', 'sun', 0, 10)}
                {slider('Sky', 'sky', 0, 2)}
                {slider('Exposure', 'exposure', 0.1, 2)}
            </Panel>
        </>
    );
}
